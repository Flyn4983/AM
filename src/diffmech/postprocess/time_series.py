"""Time-series output helpers (animation export + history I/O).

Two utilities:

- :class:`TimeSeriesWriter` — wraps a :class:`~diffmech.postprocess.vtk.PVDSeries`
  plus an in-memory history log, so a time-stepping simulation can
  simultaneously produce a ParaView animation and a NumPy ``.npz`` history
  archive.
- :func:`save_history_npz` / :func:`load_history_npz` — write a dict of
  time-aligned arrays (e.g. ``{"time": t, "u": U_history, "energy": E}``)
  to a portable ``.npz`` file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from diffmech.core import Mesh
from diffmech.postprocess.vtk import (
    FieldCollection, PVDSeries, write_vtu,
)


# ---------------------------------------------------------------------------
# History storage
# ---------------------------------------------------------------------------
class HistoryLogger:
    """In-memory time-history log.

    Append named arrays at each time step, then dump them all at the end.

    Usage
    -----
    ::

        log = HistoryLogger()
        for t in times:
            ...
            log.append(t, u=U, energy=E)
        log.save("run.npz")
    """

    def __init__(self):
        self._times: list[float] = []
        self._fields: dict[str, list[np.ndarray]] = {}

    def append(self, time: float, **fields):
        self._times.append(float(time))
        for name, value in fields.items():
            arr = np.asarray(value)
            if name not in self._fields:
                self._fields[name] = []
            self._fields[name].append(arr)

    def save(self, path: str | Path) -> None:
        """Write the full history to a NumPy ``.npz`` file."""
        out = {"time": np.asarray(self._times)}
        for name, arrs in self._fields.items():
            try:
                out[name] = np.stack(arrs, axis=0)
            except ValueError:
                # Arrays of different shapes — store as object array.
                out[name] = np.array(arrs, dtype=object)
        np.savez(str(path), **out)

    def get(self, name: str) -> np.ndarray:
        return np.stack(self._fields[name], axis=0)

    def __len__(self):
        return len(self._times)


def load_history_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a history archive saved by :class:`HistoryLogger` or :func:`save_history_npz`."""
    with np.load(str(path), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def save_history_npz(path: str | Path, **fields) -> None:
    """Convenience writer: save a dict of aligned arrays to ``.npz``."""
    np.savez(str(path), **{k: np.asarray(v) for k, v in fields.items()})


# ---------------------------------------------------------------------------
# Combined VTU animation + history logger
# ---------------------------------------------------------------------------
@dataclass
class TimeSeriesWriter:
    """Write a ParaView time animation *and* a NumPy history log.

    Attributes
    ----------
    name : base name for output files (``name.pvd``, ``name_000000.vtu``, …)
    directory : output directory
    save_every : write a VTU frame every ``save_every`` steps (1 = every step)
    fields_to_log : names of point-data fields to also store in the .npz log
                    (defaults to all fields passed to :meth:`write`).

    Example
    -------
    ::

        with TimeSeriesWriter("bending", directory="out") as w:
            for i, t in enumerate(times):
                U = solve(...)
                fields = FieldCollection(
                    point_data={"displacement": U.reshape(n, dim)},
                )
                w.write(t, mesh, fields, u=U)
        # out/bending.pvd and out/bending.npz now exist
    """

    name: str = "simulation"
    directory: str | Path = "."
    save_every: int = 1
    fields_to_log: tuple[str, ...] | None = None

    _series: PVDSeries = field(init=False, default=None)
    _logger: HistoryLogger = field(init=False, default=None)
    _step: int = field(init=False, default=0)

    def __post_init__(self):
        self._series = PVDSeries(self.name, directory=self.directory)
        self._logger = HistoryLogger()
        self._step = 0

    def write(
        self,
        time: float,
        mesh: Mesh,
        fields: FieldCollection | None = None,
        **log_values,
    ) -> None:
        """Write one frame at simulation time ``time``.

        Parameters
        ----------
        time : simulation time
        mesh : current (deformed or reference) mesh
        fields : per-node / per-cell fields to attach to the VTU
        **log_values : additional named arrays to record in the .npz history
            (typically low-dimensional scalars like ``energy=...`` or the
            full displacement vector ``u=U``).
        """
        if self._step % self.save_every == 0:
            self._series.write(time, mesh, fields)
        self._logger.append(time, **log_values)
        self._step += 1

    def close(self) -> Path:
        """Finalise the PVD master file and write the .npz history log.

        Returns the path to the ``.pvd`` file.
        """
        pvd_path = self._series.close()
        npz_path = Path(self.directory) / f"{self.name}.npz"
        self._logger.save(npz_path)
        return pvd_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        return False


# ---------------------------------------------------------------------------
# Quick single-frame writer
# ---------------------------------------------------------------------------
def save_snapshot(
    path: str | Path,
    mesh: Mesh,
    fields: FieldCollection | None = None,
    *,
    deformed: bool = False,
    displacement_field: str | None = None,
) -> None:
    """Save a single simulation snapshot as a VTU (alias for :func:`write_vtu`)."""
    write_vtu(path, mesh, fields, deformed=deformed,
              displacement_field=displacement_field)
