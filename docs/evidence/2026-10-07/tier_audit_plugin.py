"""pytest 插件（证据采集，不改被测代码）：记录每一次 ``chain_schedule`` 定价的
网格规模 / 步数 / voxel-step 代价，用于审计"哪些夹具跑不进链条档闸门"。

用法（CPU-only）::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:docs/evidence/2026-10-07 \\
      python -m pytest -q -p tier_audit_plugin

产出 ``docs/evidence/2026-10-07/tier_audit.csv``：一行一次定价调用。
"""
import csv
import os

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_audit.csv")
_rows = []
_seen = {}


def _wrap(orig):
    def patched(geometry, process, **kw):
        sched = orig(geometry, process, **kw)
        nvox = int(geometry.sdf.size)
        row = {
            "test": os.environ.get("TIER_AUDIT_TEST", ""),
            "shape": "x".join(str(s) for s in geometry.shape),
            "dx_um": round(float(geometry.spacing) * 1e6, 2),
            "n_steps": sched["n_steps"],
            "nvox": nvox,
            "voxel_steps": sched["n_steps"] * nvox,
            "t_bound_s": sched["exposure_bound_s"],
            "path_bound_mm": round(sched["path_bound_m"] * 1e3, 3),
            "v_lo": round(sched["bounds"]["scan_speed"][0], 4),
            "h_lo_um": round(sched["bounds"]["hatch_spacing"][0] * 1e6, 1),
            "lt_lo_um": round(sched["bounds"]["layer_thickness"][0] * 1e6, 1),
            "r_lo_um": round(sched["bounds"]["beam_radius"][0] * 1e6, 1),
            "policy": kw.get("resolution_policy", "strict"),
            "over_budget": kw.get("over_budget", "raise"),
        }
        key = (row["shape"], row["dx_um"], row["n_steps"], row["v_lo"])
        if key not in _seen:
            _seen[key] = True
            _rows.append(row)
        return sched
    return patched


def pytest_configure(config):
    import amforge.thermal_enthalpy as te
    import amforge.inverse as inv
    wrapped = _wrap(te.chain_schedule)      # 只包一层，两处引用同一个包装器
    te.chain_schedule = wrapped
    inv.chain_schedule = wrapped


def pytest_runtest_setup(item):
    os.environ["TIER_AUDIT_TEST"] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    if not _rows:
        return
    with open(_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(_rows, key=lambda r: -r["voxel_steps"]))
