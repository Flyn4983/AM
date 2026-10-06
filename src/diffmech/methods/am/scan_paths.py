"""Scan-path generation for AM laser trajectories.

Given a per-layer cross-section (a 2-D occupancy mask on a grid), produce the
sequence of ``(x, y)`` waypoints the laser follows to fill that layer.

Strategies:
- :func:`zigzag_hatch` — back-and-forth raster fill (the SLM workhorse)
- :func:`contour_hatch` — outline first, then zigzag fill the interior
- :func:`spiral_hatch` — Archimedean spiral (used for some LSF cladding passes)

Paths are computed at *setup time* (numpy) and returned as a flat
``(n_points, 2)`` array of waypoints, together with a per-segment velocity.
The differentiable thermal simulation then evaluates the laser position at
time ``t`` by interpolating along these waypoints (see
:mod:`diffmech.methods.am.am_thermal`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScanPath:
    """A laser trajectory on one layer.

    Attributes
    ----------
    waypoints : (n_pts, 2) array of (x, y) waypoints [m].
    segment_lengths : (n_pts-1,) distance between consecutive waypoints.
    cum_length : (n_pts,) cumulative arc length at each waypoint.
    total_length : scalar total path length.
    layer_z : z-height of this layer.
    """

    waypoints: np.ndarray
    segment_lengths: np.ndarray
    cum_length: np.ndarray
    total_length: float
    layer_z: float

    def position_at(self, s: np.ndarray) -> np.ndarray:
        """Laser position at arc-length ``s`` along the path (linear interp)."""
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.total_length)
        # Find segment index.
        idx = np.searchsorted(self.cum_length, s, side="right") - 1
        idx = np.clip(idx, 0, len(self.waypoints) - 2)
        s0 = self.cum_length[idx]
        seg_len = self.segment_lengths[idx]
        # Avoid divide-by-zero for coincident waypoints.
        t = np.where(seg_len > 1e-12, (s - s0) / np.maximum(seg_len, 1e-12), 0.0)
        p0 = self.waypoints[idx]
        p1 = self.waypoints[idx + 1]
        return p0 + t[..., None] * (p1 - p0)


def _mask_to_polygons(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                      level: float = 0.5):
    """Extract boundary polygons of a 2-D mask via marching squares."""
    try:
        from skimage import measure
    except ImportError:
        # Fall back to a convex-hull approximation if scikit-image is absent.
        from scipy.spatial import ConvexHull
        inside = np.argwhere(mask > level)
        if len(inside) < 3:
            return []
        pts = np.stack([xs[inside[:, 0]], ys[inside[:, 1]]], axis=1)
        hull = ConvexHull(pts)
        return [pts[hull.vertices]]
    polys = measure.find_contours(mask.T, level)
    # Convert from pixel indices to physical coords.
    out = []
    for p in polys:
        # p is (n, 2) in (row, col) = (y_idx, x_idx)
        px = np.interp(p[:, 1], np.arange(len(xs)), xs)
        py = np.interp(p[:, 0], np.arange(len(ys)), ys)
        out.append(np.stack([px, py], axis=1))
    return out


def zigzag_hatch(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                 *, hatch_spacing: float, direction: int = 0,
                 scan_speed: float = 1.0, layer_z: float = 0.0,
                 trim_margin: float = 0.0) -> ScanPath:
    """Generate a back-and-forth raster fill of one layer.

    Parameters
    ----------
    mask : (nx, ny) occupancy in [0, 1] on the layer.
    xs, ys : 1-D coordinate arrays of the mask grid.
    hatch_spacing : distance between adjacent hatch lines [m].
    direction : 0 = scan along x, 1 = scan along y.
    scan_speed : laser travel speed along each hatch line [m/s].
    layer_z : build height of this layer.
    trim_margin : inset the boundary by this distance (avoid edge defects).
    """
    mask = np.asarray(mask)
    if direction == 1:
        mask = mask.T
        xs, ys = ys, xs

    waypoints = []
    # Hatch lines spaced by hatch_spacing along the cross-axis.
    y_min, y_max = ys.min(), ys.max()
    n_lines = max(1, int(np.ceil((y_max - y_min) / hatch_spacing)))
    hatch_ys = np.linspace(y_min, y_max, n_lines)

    for i, yc in enumerate(hatch_ys):
        # Find x-extent of the part at this y (mask column).
        j = int(np.clip(np.argmin(np.abs(ys - yc)), 0, mask.shape[1] - 1))
        col = mask[:, j]
        inside = np.where(col > 0.5)[0]
        if len(inside) == 0:
            continue
        x_lo = xs[inside.min()]
        x_hi = xs[inside.max()]
        if trim_margin > 0:
            x_lo += trim_margin
            x_hi -= trim_margin
            if x_hi <= x_lo:
                continue
        # Alternate scan direction (zigzag).
        if i % 2 == 0:
            waypoints.append([x_lo, yc])
            waypoints.append([x_hi, yc])
        else:
            waypoints.append([x_hi, yc])
            waypoints.append([x_lo, yc])
        # Add a short jump to the next line (laser off — we keep the point
        # so the trajectory is continuous, but segment length encodes it).
        waypoints.append([waypoints[-1][0], hatch_ys[i + 1] if i + 1 < n_lines
                         else yc])

    if len(waypoints) < 2:
        waypoints = [[xs.mean(), ys.mean()]] * 2
    wp = np.asarray(waypoints, dtype=np.float64)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return ScanPath(waypoints=wp, segment_lengths=seg, cum_length=cum,
                    total_length=float(cum[-1]), layer_z=float(layer_z))


def contour_hatch(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                  *, hatch_spacing: float, contour_offset: float = 0.0,
                  scan_speed: float = 1.0, layer_z: float = 0.0) -> ScanPath:
    """Contour-then-fill strategy: outline the boundary, then zigzag fill."""
    # 1) Contour pass (boundary polygons).
    contours = _mask_to_polygons(mask, xs, ys)
    contour_pts = []
    for poly in contours:
        contour_pts.extend(poly.tolist())
        contour_pts.append(poly[0].tolist())  # close the loop

    # 2) Interior zigzag fill.
    fill = zigzag_hatch(mask, xs, ys, hatch_spacing=hatch_spacing,
                       direction=0, scan_speed=scan_speed, layer_z=layer_z,
                       trim_margin=contour_offset)

    if not contour_pts:
        return fill

    wp = np.asarray(contour_pts + fill.waypoints.tolist(), dtype=np.float64)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return ScanPath(waypoints=wp, segment_lengths=seg, cum_length=cum,
                    total_length=float(cum[-1]), layer_z=float(layer_z))


def spiral_hatch(center, radius: float, *, n_turns: float = 3.0,
                 n_points: int = 200, layer_z: float = 0.0) -> ScanPath:
    """Archimedean spiral from the centre outward (LSF cladding passes)."""
    c = np.asarray(center, dtype=np.float64)
    theta = np.linspace(0.0, n_turns * 2.0 * np.pi, n_points)
    r = np.linspace(0.0, radius, n_points)
    wp = np.stack([c[0] + r * np.cos(theta), c[1] + r * np.sin(theta)], axis=1)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return ScanPath(waypoints=wp, segment_lengths=seg, cum_length=cum,
                    total_length=float(cum[-1]), layer_z=float(layer_z))


# ---------------------------------------------------------------------------
# User-specified laser paths — the key entry point for arbitrary toolpaths
# ---------------------------------------------------------------------------
def _waypoints_to_scanpath(waypoints: np.ndarray, layer_z: float) -> ScanPath:
    """Pack a (n, 2) or (n, 3) waypoint array into a :class:`ScanPath`.

    If 3-D waypoints are supplied, the ``z`` coordinate is used as ``layer_z``
    only when ``layer_z`` is None — otherwise the provided ``layer_z`` wins.
    Duplicate consecutive points are collapsed (zero-length segments).
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2 or wp.shape[1] not in (2, 3):
        raise ValueError(
            f"waypoints must have shape (n, 2) or (n, 3); got {wp.shape}")
    if wp.shape[1] == 3:
        # Honour per-point z if no explicit layer_z was provided.
        z_used = float(wp[0, 2]) if layer_z is None else float(layer_z)
        wp = wp[:, :2]
    else:
        z_used = float(layer_z) if layer_z is not None else 0.0
    # Collapse duplicate consecutive points (zero-length segments break the
    # downstream arc-length interpolation).
    if len(wp) > 1:
        keep = np.concatenate([[True], np.linalg.norm(
            np.diff(wp, axis=0), axis=1) > 1e-12])
        wp = wp[keep]
    if len(wp) < 2:
        wp = np.tile(wp[:1] if len(wp) else np.zeros((1, 2)), (2, 1))
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return ScanPath(waypoints=wp, segment_lengths=seg, cum_length=cum,
                    total_length=float(cum[-1]), layer_z=z_used)


def from_waypoints(waypoints, *, layer_z: float | None = None) -> ScanPath:
    """Build a :class:`ScanPath` from a user-supplied waypoint array.

    This is the principal entry point for *arbitrary, externally-defined*
    laser trajectories — e.g. toolpaths exported by a slicer (Cura, Slic3r,
    IdeaMaker), an ABAQUS Python script, or a hand-written trajectory.

    Parameters
    ----------
    waypoints : (n, 2) or (n, 3) array-like
        Ordered sequence of laser ``(x, y)`` (or ``(x, y, z)``) positions [m].
        The laser visits them in order at constant scan speed (set on the
        process config). Consecutive duplicate points are collapsed.
    layer_z : float or None
        Build height of this layer [m]. If ``None`` (default):

        * for ``(n, 3)`` waypoints, the z of the first waypoint is used;
        * for ``(n, 2)`` waypoints, ``layer_z`` defaults to ``0.0``.

    Returns
    -------
    path : ScanPath
        A scan path with ``total_length``, ``segment_lengths`` and
        ``cum_length`` populated, ready to be plugged into
        :func:`multi_layer_paths` and :func:`setup_am_thermal`.

    Examples
    --------
    >>> import numpy as np
    >>> from diffmech.methods.am import from_waypoints, multi_layer_paths
    >>> # A triangle loop on layer 0, a square loop on layer 1.
    >>> tri = np.array([[0., 0.], [1., 0.], [0.5, 0.8], [0., 0.]])
    >>> sq  = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]])
    >>> mlp = multi_layer_paths([from_waypoints(tri, layer_z=0.0),
    ...                          from_waypoints(sq,  layer_z=0.5e-3)])
    """
    return _waypoints_to_scanpath(waypoints, layer_z)


def chain_paths(*paths: ScanPath, layer_z: float | None = None) -> ScanPath:
    """Concatenate several :class:`ScanPath` objects into one per-layer path.

    Useful when a single layer's toolpath mixes strategies — e.g. a contour
    pass followed by an interior zigzag fill, or two disconnected islands
    of infill joined by a travel move.

    The resulting path inherits ``layer_z`` from the first segment unless
    overridden. The connecting "travel" between the end of one segment and
    the start of the next is included as an ordinary (laser-on) segment; if
    the segments are already contiguous this contributes zero length.
    """
    if not paths:
        raise ValueError("chain_paths requires at least one path")
    z = float(layer_z) if layer_z is not None else float(paths[0].layer_z)
    wp = [np.asarray(paths[0].waypoints)]
    for p in paths[1:]:
        wp.append(np.asarray(p.waypoints))
    wp_all = np.concatenate(wp, axis=0)
    return _waypoints_to_scanpath(wp_all, z)


def from_csv(path, *, layer_z: float | None = None,
             x_col: int = 0, y_col: int = 1, skip_header: int = 1,
             delimiter: str = ",") -> ScanPath:
    """Load a laser trajectory from a CSV/text file of ``(x, y)`` columns.

    The file is parsed with :func:`numpy.loadtxt` — any whitespace- or
    comma-separated two-column layout works. Lines starting with ``#`` are
    ignored automatically by ``loadtxt``.
    """
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skip_header)
    xy = data[:, [x_col, y_col]]
    return _waypoints_to_scanpath(xy, layer_z)


def from_gcode(path, *, layer_z: float | None = None,
               feedrate: float = 1.0) -> ScanPath:
    """Parse a minimal subset of G-code into a :class:`ScanPath`.

    Recognises the common laser/printer moves:

    - ``G0`` / ``G00`` — rapid travel (laser OFF, included as a segment so
      the trajectory stays continuous; set ``feedrate`` very high to model
      near-instantaneous travel).
    - ``G1`` / ``G01`` — linear feed move (laser ON).
    - ``X.. Y.. Z.. F..`` — coordinate updates (only X/Y are consumed here;
      Z changes between layers should be split into separate paths).
    - ``;`` and ``()`` — comments.

    Only absolute coordinates (``G90``) are supported; relative (``G91``)
    raises ``NotImplementedError`` to avoid silent mis-scaling. Coordinates
    are assumed to be in millimetres and are converted to metres.

    Parameters
    ----------
    path : str or Path
        G-code text file (e.g. exported from a slicer).
    layer_z : float
        Build height of this layer [m].
    feedrate : float
        Default scan speed [m/s] used only to flag travel vs cut moves; the
        actual speed used in the simulation comes from the process config.
    """
    from pathlib import Path
    p = Path(path)
    absolute = True
    cur = np.zeros(3)
    wp: list[np.ndarray] = []
    travel_mask: list[bool] = []
    for raw in p.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("("):
            continue
        if line.startswith("G90"):
            absolute = True
            continue
        if line.startswith("G91"):
            raise NotImplementedError(
                "from_gcode currently supports only absolute (G90) coords")
        if not (line.startswith("G0") or line.startswith("G1")
                or line.startswith("G00") or line.startswith("G01")):
            continue
        is_travel = line.startswith("G0") and not line.startswith("G01")
        # Parse X/Y/Z/F tokens.
        new = cur.copy()
        for tok in line.split():
            if tok[0] in "XxYyZz":
                try:
                    val = float(tok[1:]) * 1e-3  # mm → m
                except ValueError:
                    continue
                idx = "XxYyZz".index(tok[0]) // 2
                new[idx] = val
            elif tok[0] in "Ff":
                # Feedrate token: ignored here (used in process config).
                pass
        cur = new if absolute else cur + new
        wp.append(cur[:2].copy())
        travel_mask.append(is_travel)
    if not wp:
        raise ValueError(f"no G0/G1 moves found in {path}")
    # Drop leading duplicates / collinear travel-then-cut at same point.
    wp_arr = np.asarray(wp)
    sp = _waypoints_to_scanpath(wp_arr, layer_z)
    return sp


def multi_layer_paths(paths: list[ScanPath]) -> "MultiLayerPath":
    """Bundle per-layer paths into a single trajectory with layer offsets."""
    return MultiLayerPath(paths)


@dataclass
class MultiLayerPath:
    """A sequence of per-layer scan paths stacked in build order."""

    layers: list

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def layer_heights(self) -> np.ndarray:
        return np.asarray([p.layer_z for p in self.layers])

    def total_length(self) -> float:
        return float(sum(p.total_length for p in self.layers))


__all__ = [
    "ScanPath", "MultiLayerPath", "multi_layer_paths",
    "zigzag_hatch", "contour_hatch", "spiral_hatch",
    # user-specified laser trajectories
    "from_waypoints", "chain_paths", "from_csv", "from_gcode",
]
