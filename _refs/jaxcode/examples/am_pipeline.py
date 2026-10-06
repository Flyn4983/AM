"""End-to-end differentiable AM digital-verification pipeline.

This example demonstrates the full "process → microstructure → constitutive
→ structural response → service performance" chain that DiffMech targets for
**additive-manufacturing (AM) digital verification**, with every stage written
so that ``jax.grad`` propagates from the process parameter all the way to the
final service-performance metric.

Pipeline stages
---------------

0. **Process → thermal history** (FVM heat diffusion + Gaussian laser source).
   The AM *laser power* ``P`` and *scan speed* ``v`` drive a moving Gaussian
   heat source on a 2-D FVM grid. The temperature field ``T(x, t)`` evolves
   and the *peak cooling rate* ``Ṫ`` (maximum ``−dT/dt`` after the laser
   passes) is extracted — this is the physical process parameter that
   controls grain growth.

1. **Thermal history → microstructure** (phase field, Allen-Cahn).
   The cooling rate controls the grain-growth mobility ``M``: fast cooling
   freezes a fine-grained microstructure, slow cooling yields coarse grains.
   We evolve an Allen-Cahn order parameter ``φ`` and extract a scalar
   *grain-refinement metric* ``G = ⟨|∇φ|⟩`` (interface-area density — larger
   means finer grains).

2. **Microstructure → constitutive** (Hall-Petch → CPFE).
   The Hall-Petch relation maps the grain metric to the initial critical
   resolved shear stress ``g0`` of a crystal-plasticity model::

       g0 = g0_base + k_hp / sqrt(G)

   A 2-D double-slip crystal-plasticity FEM simulation under uniaxial tension
   then yields the macroscopic equivalent plastic strain ``ε̄^p`` — the
   constitutive response of the as-built material.

3. **Constitutive → structural response** (linear-elastic FEM).
   The homogenised elastic/plastic response drives a structural FEM solve
   (cantilever bending); the resulting displacement field ``u`` is the
   structural response under service load.

4. **Structural response → service performance** (phase-field fracture).
   The FEM displacement is fed into a phase-field fracture model on a small
   Cartesian grid; the resulting *fracture energy* ``E_f`` is the
   service-performance metric (lower ``E_f`` ⇒ earlier failure ⇒ worse
   service performance).

5. **Inverse design** (``jax.grad``).
   Because every stage is differentiable, we differentiate the whole pipeline
   ``E_f(laser_power)`` with respect to the AM laser power and run a few
   gradient-descent steps to find the power that *maximises* the service
   performance (i.e. minimises the fracture-energy metric).

Run
---
::

    python examples/am_pipeline.py

Outputs are written to ``am_pipeline_out/``:
    - ``thermal_history.vtu``      final temperature field
    - ``microstructure.vtu``       phase-field order parameter
    - ``structural_response.vtu``  FEM displacement field
    - ``fracture.vtu``             phase-field fracture snapshots
    - ``am_pipeline.npz``          history log (loss vs. optimisation step)
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from diffmech.core import rectangular_mesh2d
from diffmech.materials import LinearElasticIsotropic
from diffmech.methods.fem import solve_linear_elastic
from diffmech.methods.fvm import (
    CartesianGrid, cartesian_grid_2d,
    ThermalConfig, step_thermal_scan, diffusion_dt,
    gaussian_heat_source, moving_gaussian_source,
)
from diffmech.methods.phase_field import (
    AllenCahnConfig, AllenCahnState,
    step_allen_cahn_scan, free_energy_allen_cahn,
    FractureConfig, make_fracture_state, step_fracture_scan, fracture_energy,
)
from diffmech.methods.cpfe import (
    CrystalPlasticity, CPFEProblem, solve_cpfe,
    average_equivalent_plastic_strain, double_slip_2d,
)
from diffmech.preprocess import (
    structured_quad_mesh_2d, uniaxial_tension, bending, gravity,
)
from diffmech.postprocess import (
    FieldCollection, write_vtu, TimeSeriesWriter,
)
from diffmech.solvers import DirichletBC


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent.parent / "am_pipeline_out"


# ===========================================================================
# Stage 0: Process parameter → thermal history (FVM heat diffusion)
# ===========================================================================
def thermal_history_from_laser(laser_power: jnp.ndarray,
                               *, scan_speed: float = 1.0,
                               grid: int = 24, n_steps: int = 120,
                               ) -> tuple[jnp.ndarray, CartesianGrid, jnp.ndarray]:
    """Evolve a 2-D temperature field driven by a moving Gaussian laser.

    Parameters
    ----------
    laser_power : scalar
        Absorbed laser power ``P`` [W] (the differentiable process knob).
    scan_speed : float
        Beam travel speed along ``x`` [m/s].
    grid : int
        FVM cells per direction (square domain ``L = 1 m``).
    n_steps : int
        Number of explicit heat-diffusion steps.

    Returns
    -------
    cooling_rate : scalar
        Peak cooling rate ``max(-dT/dt)`` after the beam passes [K/s] — the
        physical driver of grain growth.
    grid : CartesianGrid
        The FVM grid (used downstream for visualisation).
    T_history : (n_steps+1, grid, grid, 1) array
        Full temperature history (for export & cooling-rate extraction).
    """
    L = 1.0
    g = cartesian_grid_2d(grid, grid, lx=L, ly=L)
    # Material-like properties (dimensionless but consistent): alpha = 1e-3,
    # rho_cp = 1.0 so k = alpha * rho_cp = 1e-3.
    alpha = 1e-3
    rho_cp = 1.0
    dt = diffusion_dt(g, alpha=alpha, cfl=0.4)
    # Moving Gaussian beam: starts at x=0.25, travels in +x at scan_speed.
    src = moving_gaussian_source(
        power=laser_power, absorption=1.0, beam_radius=0.06,
        start=(0.25, 0.5), velocity=(scan_speed, 0.0),
    )
    T0 = jnp.full((grid, grid, 1), 300.0)  # ambient pre-heat
    T_hist = step_thermal_scan(
        T0, g, dt, n_steps, alpha=alpha, rho_cp=rho_cp,
        source=src, bc="neumann",
        return_history=True,
    )
    # Peak cooling rate: max over space *and* time of -dT/dt. A cell heats as
    # the laser approaches and cools once it passes; the global max of the
    # negative rate captures the fastest cooling moment anywhere, which is the
    # physical driver of grain refinement.
    dTdt = (T_hist[1:] - T_hist[:-1]) / dt  # (n_steps, grid, grid, 1)
    cooling_rate = jnp.max(-dTdt)
    return cooling_rate, g, T_hist


# ===========================================================================
# Stage 1: Thermal history → microstructure (Allen-Cahn)
# ===========================================================================
def microstructure_from_cooling_rate(cooling_rate: jnp.ndarray,
                                     n_steps: int = 50,
                                     grid: int = 24) -> tuple[AllenCahnState, AllenCahnConfig]:
    """Evolve a grain microstructure controlled by the AM cooling rate.

    The cooling rate maps to the Allen-Cahn mobility::

        M = M_min + (M_max - M_min) * sigmoid(cooling_rate)

    Fast cooling (large ``cooling_rate``) → low mobility → frozen fine grains.
    Slow cooling (small ``cooling_rate``) → high mobility → coarsened grains.

    A small random perturbation of the uniform state seeds spinodal
    decomposition (mimicking nucleation from the melt pool).
    """
    # Normalise cooling rate into a logit-like control variable so the
    # downstream sigmoid is well-behaved across physical magnitudes.
    cr_norm = cooling_rate * 1e-3
    # Map cooling rate to mobility in [0.05, 1.0].
    sigmoid = 1.0 / (1.0 + jnp.exp(-cr_norm))
    # Physical convention: fast cooling = quench = LOW mobility (frozen).
    mobility = 1.0 - 0.95 * sigmoid   # cooling_rate→+∞ ⇒ M→0.05 (fine grains)

    dx = 1.0 / grid
    dt = 1e-3
    cfg = AllenCahnConfig(dx=dx, dt=dt, mobility=mobility, kappa=1e-2,
                          bc="neumann")

    key = jax.random.PRNGKey(0)
    phi0 = jax.random.uniform(key, (grid, grid),
                              minval=-0.1, maxval=0.1).astype(jnp.float64)
    state0 = AllenCahnState(phi=phi0)
    state = step_allen_cahn_scan(state0, cfg, n_steps)
    return state, cfg


def grain_refinement_metric(state: AllenCahnState, cfg: AllenCahnConfig) -> jnp.ndarray:
    """Scalar measure of grain fineness: mean interface-area density.

    ``G = ⟨|∇φ|⟩`` — larger ⇒ more interfaces ⇒ finer grains.
    """
    from diffmech.methods.phase_field import gradient
    grad_phi = gradient(state.phi, cfg.dx, state.dim, cfg.bc)
    return jnp.mean(jnp.sqrt(jnp.sum(grad_phi ** 2, axis=-1) + 1e-12))


# ===========================================================================
# Stage 2: Microstructure → constitutive (Hall-Petch → CPFE)
# ===========================================================================
def cpfe_constitutive_response(grain_metric: jnp.ndarray,
                               *, nx: int = 4, ny: int = 4) -> jnp.ndarray:
    """Run a crystal-plasticity FEM tension test; return ⟨ε̄^p⟩.

    Hall-Petch: finer grains (larger ``G``) raise the initial CRSS::

        g0 = g0_base + k_hp / sqrt(G)
    """
    g0_base = 1.0
    k_hp = 0.5
    g0 = g0_base + k_hp / jnp.sqrt(grain_metric + 1e-6)

    b, n = double_slip_2d(theta_deg=60.0)
    mat = CrystalPlasticity(
        E=1000.0, nu=0.3,
        slip_directions=b, slip_normals=n,
        gamma_dot0=0.01, m=0.5,
        g0=g0, h0=10.0, g_sat=100.0,
    )
    mesh = rectangular_mesh2d(nx=nx, ny=ny, lx=1.0, ly=1.0, cell_type="quad4")
    case = uniaxial_tension(mesh, axis=0, strain=0.01)
    problem = CPFEProblem(
        mesh=mesh, material=mat,
        dirichlet_bcs=case.dirichlet_bcs, dim=2,
    )
    load_factors = jnp.linspace(0.0, 1.0, 10)
    sol = solve_cpfe(problem, load_factors, dt=0.1, n_sub=3)
    return average_equivalent_plastic_strain(sol.state)


# ===========================================================================
# Stage 3: Constitutive → structural response (linear-elastic FEM)
# ===========================================================================
def structural_response(eps_p_bar: jnp.ndarray,
                        *, nx: int = 8, ny: int = 4) -> tuple[jnp.ndarray, object]:
    """Gravity-loaded cantilever: the plastic response degrades the effective
    stiffness, which we model as a reduced Young's modulus::

        E_eff = E0 * (1 - alpha * eps_p_bar)

    A *force*-controlled load (gravity) is used so that the resulting
    displacement depends on the (degraded) stiffness — this is what makes
    the structural response sensitive to the upstream microstructure. A
    purely displacement-controlled load would give a stiffness-independent
    displacement and thus zero gradient through this stage.

    Returns the nodal displacement vector ``U``.
    """
    E0 = 200.0
    alpha = 20.0  # plasticity-stiffness degradation coefficient
    E_eff = E0 * (1.0 - alpha * eps_p_bar)
    mat = LinearElasticIsotropic(E=E_eff, nu=0.3)
    mesh = rectangular_mesh2d(nx=nx, ny=ny, lx=4.0, ly=1.0, cell_type="quad4")
    # Gravity load (force-controlled): clamp the left face, apply gravity.
    case = gravity(mesh, g=2.0, direction=-1,
                   clamped_face_axis=0, clamped_face_side="min")
    U = solve_linear_elastic(
        mesh, mat, list(case.dirichlet_bcs),
        body_force=case.body_force, dim=2,
    )
    return U, mesh


# ===========================================================================
# Stage 4: Structural response → service performance (phase-field fracture)
# ===========================================================================
def service_performance(U: jnp.ndarray, *, grid: int = 16) -> tuple[jnp.ndarray, object]:
    """Drive a phase-field fracture model with the FEM displacement magnitude
    and return the fracture energy (lower = worse performance).

    The FEM displacement is projected onto the fracture grid as a strain
    energy density (proportional to |u|^2), which seeds the history field.
    """
    u_max = jnp.max(jnp.abs(U))
    # Build a synthetic displacement field on the fracture grid whose amplitude
    # tracks the FEM tip displacement (differentiable proxy for the load).
    xs = jnp.linspace(0.0, 1.0, grid)
    ys = jnp.linspace(0.0, 1.0, grid)
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
    # Beam-like bending profile: u_y ~ x^2 * u_max (cantilever shape).
    disp = jnp.stack([
        jnp.zeros_like(xx),
        u_max * xx ** 2,
    ], axis=-1)  # (grid, grid, 2)

    cfg = FractureConfig(dx=1.0 / grid, dt=2e-3, Gc=1.0, ell=0.08,
                          mobility=1.0, bc="neumann")
    state0 = make_fracture_state((grid, grid), dim=2)
    state0 = type(state0)(phi=state0.phi, history=state0.history,
                          displacement=disp)
    state = step_fracture_scan(state0, cfg, n_steps=30,
                                youngs_modulus=200.0, poissons_ratio=0.3)
    Ef = fracture_energy(state, cfg, youngs_modulus=200.0, poissons_ratio=0.3)
    return Ef, state


# ===========================================================================
# Full pipeline: cooling_rate → fracture_energy
# ===========================================================================
def am_pipeline(laser_power: jnp.ndarray) -> dict:
    """End-to-end differentiable AM pipeline.

    Returns a dict of intermediate results (all differentiable quantities).
    """
    cooling_rate, thermal_grid, T_hist = thermal_history_from_laser(laser_power)
    state_ac, cfg_ac = microstructure_from_cooling_rate(cooling_rate)
    G = grain_refinement_metric(state_ac, cfg_ac)
    eps_p = cpfe_constitutive_response(G)
    U, mesh = structural_response(eps_p)
    Ef, frac_state = service_performance(U)
    return {
        "laser_power": laser_power,
        "cooling_rate": cooling_rate,
        "grain_metric": G,
        "plastic_strain": eps_p,
        "fracture_energy": Ef,
        "thermal_grid": thermal_grid,
        "thermal_history": T_hist,
        "microstructure": state_ac,
        "ac_config": cfg_ac,
        "displacement": U,
        "fem_mesh": mesh,
        "fracture_state": frac_state,
    }


def loss_fn(laser_power: jnp.ndarray) -> jnp.ndarray:
    """Service-performance objective: minimise the fracture energy metric.

    Minimising ``E_f`` corresponds to *delaying* fracture — i.e. improving
    service performance. Gradients propagate back to the laser power.
    """
    return am_pipeline(laser_power)["fracture_energy"]


# ===========================================================================
# Inverse design: gradient descent on the laser power
# ===========================================================================
def optimise_laser_power(initial_power: float = 50.0,
                          n_iter: int = 40,
                          lr: float = 5.0) -> tuple[jnp.ndarray, list]:
    """Find the laser power that minimises the fracture-energy metric."""
    grad_fn = jax.grad(loss_fn)
    power = jnp.asarray(initial_power, dtype=jnp.float64)
    history = []
    for i in range(n_iter):
        loss = float(loss_fn(power))
        g = float(grad_fn(power))
        history.append((i, float(power), loss, g))
        power = power - lr * g
    return power, history


# ===========================================================================
# Visualisation / export
# ===========================================================================
def export_results(result: dict, opt_history: list, out_dir: Path = OUT_DIR):
    """Write VTU snapshots of every pipeline stage."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 0: thermal history (final temperature field on the FVM grid).
    thermal_grid = result["thermal_grid"]
    T_hist = result["thermal_history"]
    ng = thermal_grid.nx
    mesh_t = rectangular_mesh2d(nx=ng - 1, ny=ng - 1, lx=1.0, ly=1.0,
                                cell_type="quad4")
    T_final = np.asarray(T_hist[-1]).reshape(ng, ng)
    # Sample T at mesh nodes (nearest cell-centre value is fine for preview).
    fc_t = FieldCollection(point_data={
        "temperature": T_final.ravel(),
    })
    write_vtu(out_dir / "thermal_history.vtu", mesh_t, fc_t)

    # Stage 1: microstructure.
    ac_state = result["microstructure"]
    grid = ac_state.phi.shape[0]
    mesh_ac = rectangular_mesh2d(nx=grid - 1, ny=grid - 1, lx=1.0, ly=1.0,
                                 cell_type="quad4")
    fc_ac = FieldCollection(point_data={
        "order_parameter": np.asarray(ac_state.phi).ravel(),
    })
    write_vtu(out_dir / "microstructure.vtu", mesh_ac, fc_ac)

    # Stage 3: structural response (deformed cantilever).
    U = result["displacement"]
    mesh_fem = result["fem_mesh"]
    disp = np.asarray(U).reshape(mesh_fem.n_nodes, 2)
    fc_fem = FieldCollection(point_data={"displacement": disp})
    write_vtu(out_dir / "structural_response.vtu", mesh_fem, fc_fem, deformed=True)

    # Stage 4: fracture.
    frac = result["fracture_state"]
    grid_f = frac.phi.shape[0]
    mesh_f = rectangular_mesh2d(nx=grid_f - 1, ny=grid_f - 1, lx=1.0, ly=1.0,
                                cell_type="quad4")
    fc_f = FieldCollection(point_data={
        "phase_field": np.asarray(frac.phi).ravel(),
        "history": np.asarray(frac.history).ravel(),
    })
    write_vtu(out_dir / "fracture.vtu", mesh_f, fc_f)

    # Optimisation history.
    hist = np.array([[i, r, l, g] for (i, r, l, g) in opt_history])
    np.savez(out_dir / "am_pipeline.npz",
             opt_step=hist[:, 0], laser_power=hist[:, 1],
             fracture_energy=hist[:, 2], gradient=hist[:, 3])
    return out_dir


# ===========================================================================
# Main entry point
# ===========================================================================
def main():
    print("DiffMech AM digital-verification pipeline")
    print("=" * 60)

    # Forward pass at the nominal laser power.
    print("\n[1] Forward pipeline (laser_power = 50.0 W):")
    result = am_pipeline(jnp.array(50.0))
    print(f"    peak cooling rate Ṫ    = {float(result['cooling_rate']):.4e} K/s")
    print(f"    grain metric G          = {float(result['grain_metric']):.4f}")
    print(f"    plastic strain ⟨ε̄^p⟩    = {float(result['plastic_strain']):.4e}")
    print(f"    fracture energy E_f     = {float(result['fracture_energy']):.4e}")

    # Verify end-to-end differentiability.
    print("\n[2] End-to-end gradient dE_f/d(laser_power):")
    g = float(jax.grad(loss_fn)(jnp.array(50.0)))
    print(f"    gradient = {g:.4e}")

    # Inverse design.
    print("\n[3] Inverse design (gradient descent on laser power):")
    power_opt, history = optimise_laser_power(initial_power=50.0,
                                              n_iter=40, lr=5.0)
    print(f"    initial laser power = {history[0][1]:+.4f} W, "
          f"E_f = {history[0][2]:.6e}")
    print(f"    final   laser power = {float(power_opt):+.4f} W, "
          f"E_f = {history[-1][2]:.6e}")
    print(f"    ΔE_f = {history[-1][2] - history[0][2]:+.3e} "
          f"({(history[-1][2] - history[0][2])/history[0][2]*100:+.3f}%)")

    # Export.
    print("\n[4] Exporting results to am_pipeline_out/ ...")
    out = export_results(result, history)
    print(f"    written to {out}")
    print("\nDone. Open the .vtu files in ParaView to inspect each stage.")


if __name__ == "__main__":
    main()
