"""DiffMech / AM — FastAPI bridge for server-side simulation + gradient.

This module is optional; the GUI can call it as a standalone backend, or it
can be embedded in any service. Install with ``pip install fastapi uvicorn``
then run ``uvicorn server:app --host 0.0.0.0 --port 8000 --reload``.

Endpoints
---------
* ``POST /am/session`` — create a new simulation session (geometry + method +
  process cfg + waypoints).  Returns a session id.
* ``POST /am/session/{sid}/step`` — run ``count`` steps and stream per-step
  ``fields``/``metrics`` payloads via Server-Sent Events (SSE).  Per-step
  solver state is held server-side.
* ``POST /am/session/{sid}/grad`` — run ``n_steps`` end-to-end, compute and
  return the composite quality loss :math:`\\mathcal{L}` and gradients
  :math:`d\\mathcal{L}/d(\\theta)` for the 5 tunable process variables.
* ``DELETE /am/session/{sid}`` — release the session.
"""
from __future__ import annotations

import gc
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

jax.config.update("jax_enable_x64", True)

from diffmech.methods.am import (  # noqa: E402
    SLMConfig,
    LSFConfig,
    MultiLayerPath,
    ParticleAMState,
    ParticleAMProblem,
    setup_particle_am,
    sdf_circle_2d,
    sdf_box_2d,
    sdf_from_polygon_2d,
    sdf_from_stl_mesh_3d,
    place_particles_in_sdf,
    from_waypoints,
    multi_layer_paths,
    am_loss_and_grads,
    step_am_server,
    am_api,
)
from diffmech.methods.dem import DEMConfig, make_dem_state  # noqa: E402
from diffmech.methods.mpm import MPMConfig, make_mpm_state  # noqa: E402
from diffmech.methods.sph import SPHConfig, make_sph_state  # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="DiffMech AM Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic request schemas
# ---------------------------------------------------------------------------
class GeometryRequest(BaseModel):
    """How to build the particle set for one AM build."""

    kind: str = Field(
        default="polygon_2d",
        description="one of: polygon_2d, box_2d, circle_2d, box_3d, stl_mesh_3d",
    )
    # polygon_2d
    polygon_vertices: list[list[float]] | None = Field(default=None)
    # box_2d / box_3d
    bbox: list[list[float]] | None = Field(default=None)
    # circle_2d
    circle_center: list[float] | None = Field(default=None)
    circle_radius: float | None = Field(default=None)
    # stl_mesh_3d  — base64 content of binary/ASCII STL file
    stl_base64: str | None = Field(default=None, description="base64-encoded STL file content")
    stl_scale_to: list[float] | None = Field(default=[0.2, 0.2, 0.08], description="target bounding extents [m] for Lx,Ly,Lz")
    stl_center: bool = Field(default=True)
    # layer params
    layer_zs: list[float] = Field(default_factory=lambda: [0.1, 0.3])
    spacing: float = Field(default=0.04, description="particle spacing [m]")
    dim: int = Field(default=2, ge=2, le=3)


class LayerWaypoints(BaseModel):
    layer_z: float
    waypoints: list[list[float]]  # (M, 2)


class ProcessCfg(BaseModel):
    """Unified process configuration.

    Supports both SLM (powder bed) and LSF/Clad (DED / continuous powder feed)
    by selecting ``kind``.  Fields from the *other* kind are simply ignored
    when building the typed dataclass config below.
    """

    kind: str = Field(default="SLM", pattern="^(SLM|LSF|Clad)$")
    # Common
    laser_power: float = 200.0
    scan_speed: float = 1.0
    layer_thickness: float = 40e-6
    beam_radius: float = 50e-6
    absorption: float = 0.5
    preheat_temp: float = 373.0
    # SLM / hatch
    hatch_spacing: float = 80e-6
    contour_first: bool = True
    rotation_per_layer: float = 1.5708
    # LSF / Clad specific
    track_spacing: float = 1.5e-3
    powder_feed_rate: float = 5e-4
    deposition_efficiency: float = 0.7


class SolverCfg(BaseModel):
    method: str = Field(default="dem", pattern="^(dem|mpm|sph)$")
    # DEM
    k_contact: float = 5e2
    gamma_damp: float = 0.5
    particle_radius: float = 15e-3
    particle_mass: float = 1e-9
    # MPM
    grid_h: float = 8e-3
    mpm_E: float = 193e9
    mpm_nu: float = 0.3
    rho: float = 7850.0
    # SPH
    h_smoothing: float = 30e-3


class CreateSessionRequest(BaseModel):
    geometry: GeometryRequest
    waypoints: list[LayerWaypoints]
    process: ProcessCfg = Field(default_factory=ProcessCfg)
    solver: SolverCfg = Field(default_factory=SolverCfg)
    dt: float = 1e-5
    n_steps: int = 500


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
@dataclass
class Session:
    sid: str
    method: str
    solver_state: Any
    am_state: ParticleAMState
    problem: ParticleAMProblem
    position0: jnp.ndarray
    paths: MultiLayerPath
    layer_start_times: np.ndarray
    layer_durations: np.ndarray
    cfg: SLMConfig
    solver_cfg: Any
    dt: float
    t: float = 0.0
    i: int = 0
    created_at: float = 0.0
    lock: threading.Lock | None = None


_SESSIONS: dict[str, Session] = {}
_LOCK = threading.Lock()
_SESSION_TTL = 3600  # 1h
_last_stl_info: dict = {}


def _gc_sessions() -> None:
    now = time.time()
    dead = [sid for sid, s in _SESSIONS.items() if now - s.created_at > _SESSION_TTL]
    for sid in dead:
        _SESSIONS.pop(sid, None)
    if dead:
        gc.collect()


def _build_geometry(g: GeometryRequest) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions (N, dim), layer_ids (N,))."""
    dim = g.dim
    layer_zs = np.asarray(g.layer_zs, dtype=np.float64)

    if g.kind == "polygon_2d":
        if g.polygon_vertices is None or len(g.polygon_vertices) < 3:
            raise HTTPException(400, "polygon_vertices (≥3) required for polygon_2d")
        verts = np.asarray(g.polygon_vertices, dtype=np.float64)
        sdf = sdf_from_polygon_2d(verts)
        if g.bbox is None:
            mn, mx = verts.min(axis=0), verts.max(axis=0)
            bbox = [(float(mn[d]), float(mx[d])) for d in range(verts.shape[1])]
        else:
            bbox = [(float(x[0]), float(x[1])) for x in g.bbox]
        return place_particles_in_sdf(sdf, bbox, layer_zs, dim=2, spacing=g.spacing)

    if g.kind == "box_2d":
        if g.bbox is None or len(g.bbox) != 2:
            raise HTTPException(400, "bbox=[[x0,x1],[y0,y1]] required for box_2d")
        x0, x1 = g.bbox[0]
        y0, y1 = g.bbox[1]
        bbox = [(float(x0), float(x1)), (float(y0), float(y1))]
        sdf = sdf_box_2d(np.asarray([(x0 + x1) / 2, (y0 + y1) / 2]),
                         np.asarray([x1 - x0, y1 - y0]) / 2)
        return place_particles_in_sdf(sdf, bbox, layer_zs, dim=2, spacing=g.spacing)

    if g.kind == "circle_2d":
        if g.circle_center is None or g.circle_radius is None:
            raise HTTPException(400, "circle_center + circle_radius required")
        cx, cy = g.circle_center
        r = float(g.circle_radius)
        sdf = sdf_circle_2d(np.asarray([cx, cy]), r)
        bbox = [(cx - r, cx + r), (cy - r, cy + r)]
        return place_particles_in_sdf(sdf, bbox, layer_zs, dim=2, spacing=g.spacing)

    if g.kind == "box_3d":
        if g.bbox is None or len(g.bbox) != 3:
            raise HTTPException(400, "bbox=[[x0,x1],[y0,y1],[z0,z1]] required for box_3d")
        x0, x1 = g.bbox[0]
        y0, y1 = g.bbox[1]
        z0, z1 = g.bbox[2]
        sdf = _box3d_sdf(np.asarray([(x0+x1)/2, (y0+y1)/2, (z0+z1)/2]),
                         np.asarray([x1-x0, y1-y0, z1-z0])/2)
        bbox = [(x0, x1), (y0, y1), (z0, z1)]
        return place_particles_in_sdf(sdf, bbox, layer_zs, dim=3, spacing=g.spacing)

    if g.kind == "stl_mesh_3d":
        import base64 as _b64
        if not g.stl_base64:
            raise HTTPException(400, "stl_base64 payload required for stl_mesh_3d")
        try:
            raw = _b64.b64decode(g.stl_base64, validate=True)
        except Exception as e:
            raise HTTPException(400, f"invalid stl_base64: {e}")
        try:
            sdf, info = sdf_from_stl_mesh_3d(
                raw,
                center=g.stl_center,
                scale_to=tuple(g.stl_scale_to) if g.stl_scale_to else None,
            )
        except ValueError as e:
            raise HTTPException(400, f"STL parse failed: {e}")
        bbox = [(x[0], x[1]) for x in info["bbox"]]
        # If user gave explicit layer_zs we use them; otherwise default to
        # slicing from z_min of the bbox up to z_max with particle spacing.
        if not g.layer_zs:
            z0, z1 = bbox[2]
            layers = np.arange(z0 + 0.5 * g.spacing, z1, g.spacing)
            layer_zs = layers
        pos, lids = place_particles_in_sdf(sdf, bbox, layer_zs, dim=3, spacing=g.spacing)
        # Stash mesh info on the session geometry metadata via the return path.
        _last_stl_info["info"] = info
        return pos, lids

    raise HTTPException(400, f"unknown geometry kind {g.kind!r}")


def _box3d_sdf(c: np.ndarray, half: np.ndarray):
    def _f(p: np.ndarray) -> jnp.ndarray:
        if p.ndim == 1:
            q = jnp.abs(p - c) - half
            return jnp.linalg.norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q), 0.0)
        q = jnp.abs(p - c) - half
        return jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) + jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return _f


def _build_paths(wps: list[LayerWaypoints]) -> MultiLayerPath:
    layers = []
    for w in wps:
        wp = np.asarray(w.waypoints, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[1] != 2:
            raise HTTPException(400, "waypoints must be (M, 2)")
        layers.append(from_waypoints(wp, layer_z=float(w.layer_z)))
    return multi_layer_paths(layers)


def _build_solver_state(
    s: SolverCfg, pos: np.ndarray, *, dim: int,
) -> tuple[Any, Any]:
    method = s.method
    if method == "dem":
        cfg = DEMConfig(k=float(s.k_contact), gamma=float(s.gamma_damp))
        st = make_dem_state(
            position=jnp.asarray(pos),
            radius=float(s.particle_radius),
            mass=float(s.particle_mass),
            dim=dim,
        )
        return cfg, st
    if method == "mpm":
        # Auto-grid from bbox
        p = np.asarray(pos)
        n_grid = 8
        grid_origin = tuple(float(x) for x in p.min(axis=0))
        spans = p.max(axis=0) - p.min(axis=0)
        dx = float(max(spans) / max(1, n_grid - 2))
        shape = tuple(max(3, int(np.ceil(spans[i] / dx)) + 2) for i in range(dim))
        cfg = MPMConfig(
            grid_origin=grid_origin,
            grid_shape=shape,
            dx=dx,
            dt=1e-5,
            youngs_modulus=float(s.mpm_E),
            poissons_ratio=float(s.mpm_nu),
            rho0=float(s.rho),
        )
        volume = dx ** dim
        st = make_mpm_state(
            position=jnp.asarray(pos), volume=volume,
            mass=float(s.particle_mass), dim=dim,
        )
        return cfg, st
    if method == "sph":
        cfg = SPHConfig(h=float(s.h_smoothing), rho0=float(s.rho),
                        c0=50.0, nu=0.01)
        st = make_sph_state(position=jnp.asarray(pos), mass=float(s.particle_mass))
        return cfg, st
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/am/session")
def create_session(req: CreateSessionRequest) -> dict[str, Any]:
    with _LOCK:
        _gc_sessions()

    pos, lids = _build_geometry(req.geometry)
    paths = _build_paths(req.waypoints)

    pc = req.process
    # Build either SLMConfig (powder-bed) or LSFConfig (DED / powder feed)
    # based on ProcessCfg.kind.  Clad == LSF for our purposes (same mass
    # source physics, only a semantic difference for the user).
    if pc.kind in ("LSF", "Clad"):
        cfg = LSFConfig(
            laser_power=pc.laser_power,
            scan_speed=pc.scan_speed,
            layer_thickness=pc.layer_thickness,
            track_spacing=pc.track_spacing,
            beam_radius=pc.beam_radius,
            powder_feed_rate=pc.powder_feed_rate,
            absorption=pc.absorption,
            deposition_efficiency=pc.deposition_efficiency,
            preheat_temp=pc.preheat_temp,
        )
    else:  # SLM (default)
        cfg = SLMConfig(
            laser_power=pc.laser_power,
            scan_speed=pc.scan_speed,
            layer_thickness=pc.layer_thickness,
            hatch_spacing=pc.hatch_spacing,
            beam_radius=pc.beam_radius,
            absorption=pc.absorption,
            contour_first=pc.contour_first,
            rotation_per_layer=pc.rotation_per_layer,
            preheat_temp=pc.preheat_temp,
        )
    problem = setup_particle_am(
        pos, lids, paths, cfg,
        dim=req.geometry.dim,
        particle_mass=req.solver.particle_mass,
    )
    solver_cfg, solver_state = _build_solver_state(
        req.solver, pos, dim=req.geometry.dim,
    )
    sid = uuid.uuid4().hex
    sess = Session(
        sid=sid,
        method=req.solver.method,
        solver_state=solver_state,
        am_state=problem.am_state,
        problem=problem,
        position0=jnp.asarray(pos),
        paths=paths,
        layer_start_times=np.asarray(problem.layer_start_times),
        layer_durations=np.asarray(problem.layer_durations),
        cfg=cfg,
        solver_cfg=solver_cfg,
        dt=req.dt,
        created_at=time.time(),
        lock=threading.Lock(),
    )
    with _LOCK:
        _SESSIONS[sid] = sess
    resp = {
        "session_id": sid,
        "dim": req.geometry.dim,
        "n_particles": int(pos.shape[0]),
        "n_layers": int(len(req.waypoints)),
        "dt": req.dt,
        "method": req.solver.method,
        "process_kind": pc.kind,
    }
    if req.geometry.kind == "stl_mesh_3d" and "info" in _last_stl_info:
        info = _last_stl_info["info"]
        resp["mesh"] = {
            "n_faces": int(info["n_faces"]),
            "bbox": info["bbox"],
        }
    return resp


def _get_session(sid: str) -> Session:
    with _LOCK:
        _gc_sessions()
        s = _SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "session not found")
    return s


@app.delete("/am/session/{sid}")
def delete_session(sid: str) -> dict[str, Any]:
    with _LOCK:
        s = _SESSIONS.pop(sid, None)
    if s is None:
        raise HTTPException(404, "session not found")
    return {"deleted": True, "sid": sid, "i": s.i, "t": s.t}


@app.post("/am/session/{sid}/step")
async def run_steps_stream(sid: str, count: int = 50):
    """Stream per-step ``fields``/``metrics`` updates via SSE."""
    s = _get_session(sid)
    count = int(max(1, min(count, 20000)))

    async def gen():
        with s.lock:
            for k in range(count):
                out = step_am_server(
                    s.method,
                    solver_state=s.solver_state,
                    am_state=s.am_state,
                    dt=s.dt,
                    t=s.t,
                    solver_cfg=s.solver_cfg,
                    cfg=s.cfg,
                    paths=s.paths,
                    layer_start_times=s.layer_start_times,
                    layer_durations=s.layer_durations,
                    position0=s.position0,
                )
                s.solver_state = out["solver_state"]
                s.am_state = out["am_state"]
                s.i += 1
                s.t = float(out["t_next"])
                payload = {
                    "i": s.i,
                    "t": s.t,
                    "dt": s.dt,
                    **out["json_only"],
                }
                yield {"event": "step", "data": payload}
                # Yield control to the event loop once per ~batch.
                if (k + 1) % 10 == 0:
                    import asyncio
                    await asyncio.sleep(0)
            yield {"event": "done", "data": {"i": s.i, "t": s.t}}

    return EventSourceResponse(gen())


class GradRequest(BaseModel):
    n_steps: int = 200
    process_vars: list[str] | None = None
    dt: float | None = None
    T_melt: float = 1700.0
    sigma_yield: float = 200e6
    u_max: float = 1e-3


@app.post("/am/session/{sid}/grad")
def compute_gradient(sid: str, req: GradRequest) -> dict[str, Any]:
    """End-to-end AM loss + gradients w.r.t. selected process variables."""
    s = _get_session(sid)
    with s.lock:
        dt = float(req.dt if req.dt is not None else s.dt)
        pvars = tuple(req.process_vars) if req.process_vars else (
            "laser_power", "scan_speed", "beam_radius", "absorption", "preheat_temp",
        )
        result = am_loss_and_grads(
            s.method,
            solver_state0=s.solver_state,
            solver_cfg=s.solver_cfg,
            problem0=s.problem,
            position0=s.position0,
            n_steps=int(req.n_steps),
            dt=dt,
            process_vars=pvars,
            T_melt=float(req.T_melt),
            sigma_yield=float(req.sigma_yield),
            u_max=float(req.u_max),
        )
    return result


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "jax": jax.__version__, "sessions": len(_SESSIONS)}


# ---------------------------------------------------------------------------
# P1-2  Gradient-driven parameter optimization loop (SSE stream)
# ---------------------------------------------------------------------------

class OptRequest(BaseModel):
    max_iter: int = 20
    n_steps: int = 40
    dt: float | None = None
    process_vars: list[str] | None = None
    lr: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    T_melt: float = 1700.0
    sigma_yield: float = 200e6
    u_max: float = 1e-3


def _adam_step(
    x: jnp.ndarray, m: jnp.ndarray, v: jnp.ndarray, g: jnp.ndarray, t: int,
    lr: float, beta1: float, beta2: float, eps: float,
    lo: jnp.ndarray, hi: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One Adam step with projected bounds."""
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    mhat = m / (1.0 - jnp.power(beta1, t))
    vhat = v / (1.0 - jnp.power(beta2, t))
    x = x - lr * mhat / (jnp.sqrt(vhat) + eps)
    x = jnp.clip(x, lo, hi)
    return x, m, v


@app.post("/am/session/{sid}/opt")
async def run_optimization_stream(sid: str, req: OptRequest):
    """Run gradient-based Adam optimization, streaming history over SSE."""
    from src.diffmech.methods.am.am_api import _PROCESS_BOUNDS

    s = _get_session(sid)
    dt = float(req.dt if req.dt is not None else s.dt)
    pvars = tuple(req.process_vars) if req.process_vars else (
        "laser_power", "scan_speed", "beam_radius", "absorption", "preheat_temp",
    )
    for k in pvars:
        if k not in _PROCESS_BOUNDS:
            raise HTTPException(status_code=400, detail=f"unknown process var {k!r}")

    def yield_json(d):
        return json.dumps(d, allow_nan=False, default=str)

    async def gen():
        import asyncio
        with s.lock:
            base_cfg = s.problem.cfg
            x_arr = [float(getattr(base_cfg, k)) for k in pvars]
            lo = jnp.asarray([_PROCESS_BOUNDS[k][0] for k in pvars], dtype=jnp.float64)
            hi = jnp.asarray([_PROCESS_BOUNDS[k][1] for k in pvars], dtype=jnp.float64)
            x = jnp.asarray(x_arr, dtype=jnp.float64)
            m = jnp.zeros_like(x)
            v = jnp.zeros_like(x)
        for t in range(1, int(req.max_iter) + 1):
            # Allow cancellation between iterations via async sleep.
            await asyncio.sleep(0)
            with s.lock:
                result = am_loss_and_grads(
                    s.method,
                    solver_state0=s.solver_state,
                    solver_cfg=s.solver_cfg,
                    problem0=s.problem,
                    position0=s.position0,
                    n_steps=int(req.n_steps),
                    dt=dt,
                    process_vars=pvars,
                    T_melt=float(req.T_melt),
                    sigma_yield=float(req.sigma_yield),
                    u_max=float(req.u_max),
                )
            grads = jnp.asarray([float(result["grads"].get(k, 0.0)) for k in pvars], dtype=jnp.float64)
            # Guard against NaN/inf
            grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
            with s.lock:
                x, m, v = _adam_step(
                    x, m, v, grads, t,
                    lr=float(req.lr), beta1=float(req.beta1),
                    beta2=float(req.beta2), eps=float(req.eps),
                    lo=lo, hi=hi,
                )
                nx = [float(vv) for vv in x.tolist()]
                # Apply x_new to session process cfg (persisted across runs/grads)
                merged = {k: jnp.asarray(getattr(s.problem.cfg, k)) for k in am_api._PROCESS_KEYS}
                for i, k in enumerate(pvars):
                    merged[k] = jnp.asarray(nx[i])
                new_cfg = am_api._make_cfg(s.problem.cfg, merged)
                _lens = am_api.particle_layer_path_lengths(s.problem.paths)
                lens = jnp.asarray(_lens, dtype=jnp.float64)
                durs = lens / new_cfg.scan_speed
                starts = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durs)[:-1]])
                s.problem = ParticleAMProblem(
                    am_state=s.problem.am_state,
                    paths=s.problem.paths,
                    cfg=new_cfg,
                    layer_start_times=starts,
                    layer_durations=durs,
                    dim=s.problem.dim,
                )
            frame = {
                "iter": t,
                "loss": float(result["loss"]),
                "diagnostics": result.get("diagnostics", {}),
                "param_values": {k: nx[i] for i, k in enumerate(pvars)},
                "grads": {k: float(result["grads"].get(k, 0.0)) for i, k in enumerate(pvars)},
                "bounds": {k: list(_PROCESS_BOUNDS[k]) for k in pvars},
            }
            yield {"event": "iter", "data": yield_json(frame)}
        yield {"event": "done", "data": yield_json({"max_iter": int(req.max_iter)})}

    return EventSourceResponse(gen())
