"""Lightweight GIF / MP4 animation of time-stepped fields.

ParaView (via the :class:`~diffmech.postprocess.vtk.PVDSeries` writer) is the
production tool for inspecting transient results, but a self-contained
``.gif`` is invaluable for quick sharing in chat, notebooks and reports.
This module turns a stack of 2-D scalar fields ``(n_steps, nx, ny)`` (or a
list of arrays) into an animated GIF with minimal boilerplate.

It uses matplotlib's ``FuncAnimation`` + ``Pillow`` writer, so no ffmpeg is
required. For 3-D fields, pass ``slicer`` to extract a 2-D plane.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def _mpl_agg():
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    return plt


def save_field_gif(
    fields: Sequence,
    path: str | Path,
    *,
    title: str = "",
    cmap: str = "viridis",
    extent=None,
    fps: int = 10,
    colorbar: bool = True,
    vmin=None,
    vmax=None,
    slicer: Callable[[np.ndarray], np.ndarray] | None = None,
    labels: Sequence[str] | None = None,
) -> Path:
    """Animate a sequence of 2-D scalar fields as a GIF.

    Parameters
    ----------
    fields : sequence of (nx, ny) arrays, or a single (n_steps, nx, ny) array.
        The time history of a scalar field. 3-D arrays ``(nx, ny, nz)`` are
        accepted if ``slicer`` is given to reduce them to 2-D.
    path : output ``.gif`` (or ``.mp4``) path.
    title : figure title (may include ``{i}`` for the step index).
    cmap : matplotlib colormap.
    extent : ``(xmin, xmax, ymin, ymax)`` passed to ``imshow``.
    fps : frames per second.
    colorbar : draw a colorbar.
    vmin, vmax : fixed color scale; if ``None`` auto-scaled per-frame.
    slicer : callable taking a 2-D/3-D field and returning a 2-D slice.
        e.g. ``slicer=lambda f: f[:, :, f.shape[2]//2]``.
    labels : per-frame annotation text (e.g. time stamps).

    Returns
    -------
    path : the written file path.
    """
    fields = np.asarray(fields)
    if fields.ndim == 2:
        fields = fields[None]
    if fields.ndim != 3:
        raise ValueError(
            f"fields must be 2-D (nx,ny) or 3-D (n,nx,ny); got shape {fields.shape}"
        )

    if slicer is not None:
        fields = np.stack([slicer(f) for f in fields])

    plt = _mpl_agg()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    if vmin is None:
        vmin = float(np.nanmin(fields))
    if vmax is None:
        vmax = float(np.nanmax(fields))
    im = ax.imshow(fields[0].T, origin="lower", cmap=cmap, vmin=vmin,
                   vmax=vmax, extent=extent, aspect="auto")
    if colorbar:
        fig.colorbar(im, ax=ax)
    txt = ax.set_title(title.format(i=0) if "{i}" in title else title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i):
        im.set_data(fields[i].T)
        if labels is not None and i < len(labels):
            txt.set_text(labels[i])
        elif "{i}" in title:
            txt.set_text(title.format(i=i))
        return im, txt

    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    anim = FuncAnimation(fig, update, frames=len(fields),
                         interval=1000.0 / fps, blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp4":
        anim.save(str(path), writer=FFMpegWriter(fps=fps))
    else:
        anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def save_field_grid_png(
    fields: Sequence,
    path: str | Path,
    *,
    titles: Sequence[str] | None = None,
    cmap: str = "viridis",
    ncols: int = 3,
) -> Path:
    """Save a grid of 2-D field snapshots to a single PNG (quick report figure).

    Useful for a compact "time evolution" or "parameter sweep" summary image.
    """
    fields = np.asarray(fields)
    if fields.ndim == 2:
        fields = fields[None]
    n = len(fields)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    plt = _mpl_agg()
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows),
                            squeeze=False)
    vmin = float(np.nanmin(fields))
    vmax = float(np.nanmax(fields))
    for k, ax in enumerate(axes.ravel()):
        if k < n:
            ax.imshow(fields[k].T, origin="lower", cmap=cmap,
                      vmin=vmin, vmax=vmax, aspect="auto")
            if titles is not None and k < len(titles):
                ax.set_title(titles[k], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path


__all__ = ["save_field_gif", "save_field_grid_png"]
