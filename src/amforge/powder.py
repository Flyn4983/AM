"""粒子 / 粉末床数据模型 + 导入 / 生成（缺口 #20-②B）

缺口 #20-②B：GUI 前处理「导入粒子 / 粉末」。

本模块提供**实打实**的粒子（粉末）数据模型，而不是装饰性 UI：

* :class:`ParticleCollection` 是真实的点云（逐粒子坐标 + 半径 + 材料），
  可从 **CSV / JSON 点云文件**导入（坐标 + 半径 + 可选材料），
  也可由 ``from_part_bed`` **真实生成**一个随机密堆粉末层
  （抖动晶格 + 对数正态粒径分布 PSD，与 ``diffmech`` 的 ``synth_powder_bed``
  同源构造）；
* ``packing_density`` / ``stats`` 在**导入/生成的真实粒子**上直接测量
  （体积分数、d50、PSD 标准差、材料密度），而非假设；
* ``powder_bed_params`` 把真实统计映射到 ``powder.bed`` 求解器的参数
  （``d50`` / ``psd_sigma`` / ``rho_powder``），``solve_dem`` 据此调用
  **真实的 DEM 铺粉求解** ``amforge.powderbed.solve_powderbed(...,
  powder_fidelity="dem")``（跑 ``step_dem_scan`` 重力沉降，得到真解出的
  密实度 / 配位数 / 粗糙度），返回真实 ``PowderBedResult``。

设计约束（与审计路线一致）
--------------------------
* **纯数据 + 适配层，不触碰物理内核**：DEM/SPH/MPM 接触/流体物理完全复用
  ``diffmech`` 现成内核，本模块只负责"粒子点云 ↔ 粉末床求解器"的桥接。
* **可微链路无关**：粒子点云是 GUI 前处理的导入态，不进入 ``jax.grad``
  逆向链（与 STL 几何导入同定位）；其统计参数可在正向求解时驱动真实 DEM。
* **零依赖导入**：``solve_dem`` 惰性导入 ``amforge.powderbed``（进而
  ``diffmech``），使本模块在无显示 / 轻量测试环境下也能干净 import。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from amforge.core.contracts import PartGeometry
from amforge.materials import get_material

VALID_CSV_FORMATS = "CSV: x,y,r (2D) | x,y,z,r (3D) | x,y,z,r,material (3D); JSON: {material, particles:[[x,y,z,r]…]} 或 {x,y,z,r,material}"


@dataclass(frozen=True)
class ParticleCollection:
    """粉末 / 粒子点云（逐粒子坐标 + 半径 + 材料）。

    Attributes
    ----------
    position : (N, dim) JAX 数组 [m]
        粒子中心坐标。2D 下 ``dim==2``，坐标为 ``(x, y)``；3D 下 ``(x, y, z)``。
    radius : (N,) JAX 数组 [m]
        逐粒子半径（即粒径 d = 2·r 的中位数即 d50）。
    material : str
        材料名（交给 ``get_material`` 取密度 ``rho_solid``）。
    dim : int
        2 或 3（静态）。
    name : str
        来源标签（导入文件名 / "generated bed" 等）。
    """

    position: jnp.ndarray
    radius: jnp.ndarray
    material: str = "316L"
    dim: int = 3
    name: str = "particles"

    # ---- 基本量 ----------------------------------------------------------
    @property
    def n_particles(self) -> int:
        return int(self.position.shape[0])

    def __eq__(self, other) -> bool:
        if not isinstance(other, ParticleCollection):
            return NotImplemented
        return (self.dim == other.dim and self.material == other.material
                and self.name == other.name
                and np.array_equal(np.asarray(self.position), np.asarray(other.position))
                and np.array_equal(np.asarray(self.radius), np.asarray(other.radius)))

    def _pos_np(self) -> np.ndarray:
        return np.asarray(self.position, dtype=np.float64)

    def _rad_np(self) -> np.ndarray:
        return np.asarray(self.radius, dtype=np.float64)

    # ===================================================================
    # 1. 导入：CSV / JSON 点云
    # ===================================================================
    @classmethod
    def from_csv(cls, path: str | Path, *, material: str = "316L",
                 name: str | None = None) -> "ParticleCollection":
        """从 CSV 点云导入。

        列约定（支持 ``#`` 注释行与表头行，首列非数字即视为表头跳过）：
          * 2D : ``x, y, r``
          * 3D : ``x, y, z, r``
          * 3D + 材料 : ``x, y, z, r, material``（第 5 列字符串为材料名）

        单位 SI（m）。逐粒子坐标与半径被原样保留——**导入即真实点云**。
        """
        rows: list[tuple[list[float], float, str | None]] = []
        parsed_name: str | None = None
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            # 注释行支持 `# name=...`：导出往返时保留集合名，使 roundtrip 完整
            # （允许含空格的名称，如 "generated bed"）
            if "#" in raw:
                cm = re.search(r"name=(.*)", raw.split("#", 1)[1])
                if cm:
                    parsed_name = cm.group(1).strip()
                raw = raw.split("#", 1)[0]
            line = raw.strip()
            if not line:
                continue
            toks = [t.strip() for t in line.split(",")]
            try:
                float(toks[0])
            except ValueError:
                continue  # 表头 / 非数据行
            if len(toks) == 3:
                rows.append(([float(toks[0]), float(toks[1])], float(toks[2]), None))
            elif len(toks) == 4:
                rows.append(([float(toks[0]), float(toks[1]), float(toks[2])],
                             float(toks[3]), None))
            elif len(toks) == 5:
                rows.append(([float(toks[0]), float(toks[1]), float(toks[2])],
                             float(toks[3]), toks[4]))
            else:
                raise ValueError(
                    f"CSV 列数应为 3/4/5，得到 {len(toks)}（{VALID_CSV_FORMATS}）")
        if not rows:
            raise ValueError(f"CSV 未解析到任何粒子：{path}")
        positions = [r[0] for r in rows]
        radius = np.asarray([r[1] for r in rows], dtype=np.float64)
        mats = [r[2] for r in rows if r[2] is not None]
        dim = 3 if len(positions[0]) == 3 else 2
        mat = material
        if mats:
            uniq = set(mats)
            if len(uniq) == 1:
                mat = next(iter(uniq))
            # 多材料点云：取首个材料做集合级密度（粉末床尺度默认单材料）
        return cls(
            position=jnp.asarray(np.asarray(positions, dtype=np.float64)),
            radius=jnp.asarray(radius),
            material=mat, dim=dim,
            name=name or parsed_name or Path(path).stem,
        )

    @classmethod
    def from_json(cls, path: str | Path, *,
                  name: str | None = None) -> "ParticleCollection":
        """从 JSON 点云导入。

        两种结构均支持：
          * ``{"material": "316L", "particles": [[x,y,z,r], ...]}``（3D）或
            ``[[x,y,r], ...]``（2D）；
          * ``{"x":[...], "y":[...], "z":[...]?, "r":[...], "material":"316L"}``。
        """
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        material = str(d.get("material", "316L"))
        if "particles" in d:
            arr = np.asarray(d["particles"], dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError("particles 应为二维数组 (N, 3) 或 (N, 4)")
            if arr.shape[1] == 3:
                pos = np.column_stack([arr[:, 0], arr[:, 1], np.zeros(arr.shape[0])])
                rad = arr[:, 2]
                dim = 2
            elif arr.shape[1] == 4:
                pos = arr[:, :3]
                rad = arr[:, 3]
                dim = 3
            else:
                raise ValueError("particles 列数应为 3(2D) 或 4(3D)")
        elif all(k in d for k in ("x", "y", "r")):
            x = np.asarray(d["x"], dtype=np.float64)
            y = np.asarray(d["y"], dtype=np.float64)
            r = np.asarray(d["r"], dtype=np.float64)
            z = np.asarray(d.get("z", np.zeros_like(x)), dtype=np.float64)
            if np.any(z != 0.0) or "z" in d:
                pos = np.column_stack([x, y, z]); dim = 3
            else:
                pos = np.column_stack([x, y]); dim = 2
            rad = r
        else:
            raise ValueError(
                "JSON 需含 'particles' 或 'x'/'y'/'r' 键（" + VALID_CSV_FORMATS + "）")
        return cls(
            position=jnp.asarray(pos), radius=jnp.asarray(rad),
            material=material, dim=int(dim),
            name=name if name is not None else str(d.get("name", Path(path).stem)),
        )

    # ===================================================================
    # 2. 生成：真实随机密堆粉末层（抖动晶格 + 对数正态 PSD）
    # ===================================================================
    @classmethod
    def from_part_bed(cls, part: PartGeometry, *,
                      layer_thickness: float = 40e-6,
                      d50: float = 30e-6,
                      psd_sigma: float = 0.25,
                      n_layers: int = 1,
                      seed: int = 0,
                      material: str = "316L",
                      jitter: float = 0.35) -> "ParticleCollection":
        """在零件 bbox 的构建区生成一层**真实**粉末床点云。

        构造与 ``diffmech`` 的 ``synth_powder_bed`` 同源：规则晶格（间距
        ≈2·d50，保证初态无重叠）+ **亚粒径量级抖动**（打破对称，使后续
        DEM 沉降到真实 RCP）+ **对数正态 PSD**（``d = d50·exp(σ·z)``，
        限幅到 ``[0.3, 3.0]·d50`` 避免病态接触）。

        填充区域：``x,y`` 取零件 bbox 整窗（SLM 粉末铺满成形窗口），
        ``z`` 从 bbox 底面起、向上 ``n_layers·layer_thickness`` 厚。
        这是真实数值构造的粒子集，可直接喂 DEM 评估。
        """
        lo, hi = part.bbox()
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        dz_total = float(n_layers) * float(layer_thickness)
        if part.dim == 2:
            # 2D：构建方向是第二轴（y），x 铺满窗口，y 为粉层厚度
            x0, x1 = float(lo[0]), float(hi[0])
            y0, y1 = float(lo[1]), float(lo[1]) + dz_total
            z0 = z1 = 0.0
        else:
            x0, x1 = float(lo[0]), float(hi[0])
            y0, y1 = float(lo[1]), float(hi[1])
            z0 = float(lo[2])
            z1 = z0 + dz_total

        spacing = 2.0 * d50
        nx = max(1, int(math.floor((x1 - x0) / spacing)) + 1)
        ny = max(1, int(math.floor((y1 - y0) / spacing)) + 1)
        nz = max(1, int(math.floor((z1 - z0) / spacing)) + 1) if part.dim == 3 else 1
        n = nx * ny * nz

        key = jax.random.PRNGKey(int(seed))
        k_r, k_j = jax.random.split(key)
        # 对数正态粒径
        zz = jax.random.normal(k_r, (n,), dtype=jnp.float64)
        diameter = float(d50) * jnp.exp(float(psd_sigma) * zz)
        diameter = jnp.clip(diameter, 0.3 * float(d50), 3.0 * float(d50))
        radius = 0.5 * diameter

        # 晶格坐标（形状静态 arange，乘具体间距）
        ix = jnp.arange(nx, dtype=jnp.float64)
        iy = jnp.arange(ny, dtype=jnp.float64)
        iz = jnp.arange(nz, dtype=jnp.float64)
        dx = (x1 - x0) / max(nx, 1)
        dy = (y1 - y0) / max(ny, 1)
        dz = (z1 - z0) / max(nz, 1) if nz > 1 else (z1 - z0)

        if part.dim == 3:
            gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing="ij")
            pos = jnp.stack([x0 + (gx + 0.5) * dx,
                             y0 + (gy + 0.5) * dy,
                             z0 + (gz + 0.5) * dz], axis=-1).reshape(-1, 3)
        else:
            gx, gz = jnp.meshgrid(ix, iz, indexing="ij")
            pos = jnp.stack([x0 + (gx + 0.5) * dx,
                             y0 + (gz + 0.5) * dy], axis=-1).reshape(-1, 2)

        jit = jax.random.uniform(k_j, pos.shape, dtype=jnp.float64,
                                 minval=-jitter, maxval=jitter) * float(d50)
        pos = pos + jit

        return cls(
            position=pos, radius=radius, material=material,
            dim=int(part.dim), name="generated bed",
        )

    # ===================================================================
    # 3. 导出
    # ===================================================================
    def to_csv(self, path: str | Path) -> str:
        """导出为 CSV（含材料注释头）。返回文件路径。"""
        pos = self._pos_np()
        rad = self._rad_np()
        path = Path(path)
        lines = [f"# name={self.name}",
                 f"# material={self.material}  dim={self.dim}  n={self.n_particles}",
                 f"# {VALID_CSV_FORMATS}"]
        if self.dim == 3:
            lines.append("x,y,z,r")
            for (x, y, z), rr in zip(pos, rad):
                lines.append(f"{x:.17e},{y:.17e},{z:.17e},{float(rr):.17e}")
        else:
            lines.append("x,y,r")
            for (x, y), rr in zip(pos, rad):
                lines.append(f"{x:.17e},{y:.17e},{float(rr):.17e}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def to_json(self, path: str | Path) -> str:
        """导出为 JSON 点云。返回文件路径。"""
        pos = self._pos_np()
        rad = self._rad_np()
        if self.dim == 3:
            particles = [[float(p[0]), float(p[1]), float(p[2]), float(rr)]
                         for p, rr in zip(pos, rad)]
        else:
            particles = [[float(p[0]), float(p[1]), float(rr)]
                         for p, rr in zip(pos, rad)]
        d = {"name": self.name, "material": self.material, "dim": self.dim,
             "n_particles": self.n_particles, "particles": particles}
        path = Path(path)
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        return str(path)

    # ===================================================================
    # 4. 真实统计（在导入/生成的粒子上直接测量）
    # ===================================================================
    def _bbox(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self._pos_np()
        return pos.min(axis=0), pos.max(axis=0)

    def packing_density(self, bed_lo: Sequence[float] | None = None,
                        bed_hi: Sequence[float] | None = None) -> float:
        """真实体积分数 = Σ颗粒体积 / 粉床包围盒体积（在真实粒子上测量）。

        ``bed_lo``/``bed_hi`` 缺省时取粒子自身 bbox。2D 退化为面积分数。
        """
        pos = self._pos_np()
        rad = self._rad_np()
        if bed_lo is None or bed_hi is None:
            lo, hi = self._bbox()
        else:
            lo = np.asarray(bed_lo, dtype=np.float64)
            hi = np.asarray(bed_hi, dtype=np.float64)
        ext = np.maximum(hi - lo, 1e-12)
        if self.dim == 3:
            v_p = float(np.sum((4.0 / 3.0) * np.pi * rad ** 3))
            v_box = float(np.prod(ext))
        else:
            v_p = float(np.sum(np.pi * rad ** 2))
            v_box = float(np.prod(ext[:2]))
        return float(np.clip(v_p / v_box, 0.0, 1.0))

    def stats(self) -> dict[str, float]:
        """真实粉末统计：粒子数、d50、PSD 对数标准差、平均粒径、材料密度、bbox。"""
        rad = self._rad_np()
        diam = 2.0 * rad
        d50 = float(np.median(diam))
        with np.errstate(divide="ignore"):
            ln_ratios = np.log(np.maximum(diam / d50, 1e-12))
        sigma = float(np.std(ln_ratios))
        rho = float(get_material(self.material).rho_solid)
        lo, hi = self._bbox()
        return {
            "n_particles": self.n_particles,
            "d50": d50,
            "psd_sigma": sigma,
            "mean_diameter": float(np.mean(diam)),
            "rho": rho,
            "material": self.material,
            "dim": self.dim,
            "bbox_lo": tuple(float(v) for v in lo),
            "bbox_hi": tuple(float(v) for v in hi),
        }

    def powder_bed_params(self, *, n_steps: int = 120,
                          geometry: PartGeometry | None = None,
                          use_precise: bool = False) -> dict[str, Any]:
        """把本集合的真实统计映射为 ``powder.bed`` 求解器参数。

        真实粒子统计（d50 / PSD / 材料密度）**直接驱动**真实 DEM 铺粉求解；
        粉床横向尺寸由零件 bbox 决定（``solve_powderbed`` 内部读取）。

        若给出 ``geometry``，则用「零件 bbox ÷ 2·d50」给 DEM 床定出合理的
        晶格规模（nx/ny/nz），使沉降后达到真实 RCP 量级——而非默认 6×6×3
        在大零件上退化成极稀疏粉床（密实度被床体积稀释失真）。
        """
        s = self.stats()
        d50 = s["d50"]
        params = {
            "powder_fidelity": "dem",
            "d50": d50,
            "psd_sigma": s["psd_sigma"],
            "rho_powder": s["rho"],
            "dem": {"n_steps": int(n_steps)},
        }
        if geometry is not None:
            lo, hi = np.asarray(geometry.bbox(), dtype=np.float64)
            sp = 2.0 * d50
            nx = max(4, int(math.floor((hi[0] - lo[0]) / sp)) + 1)
            if int(geometry.dim) == 2:
                ny = 1
                nz = max(2, int(math.floor((hi[1] - lo[1]) / sp)) + 1)
            else:
                ny = max(4, int(math.floor((hi[1] - lo[1]) / sp)) + 1)
                nz = 3
            params["nx"] = nx
            params["ny"] = ny
            params["nz"] = nz
        if use_precise:
            # 把导入/生成的精确粒子坐标/半径作为 DEM 初始态注入（opt-in）
            params["initial_positions"] = np.asarray(
                self.position, dtype=np.float64)
            params["initial_radii"] = np.asarray(self.radius, dtype=np.float64)
        return params

    # ===================================================================
    # 5. 真实求解：驱动 DEM 铺粉，得到 PowderBedResult
    # ===================================================================
    def solve_dem(self, geometry: PartGeometry, process, *,
                  n_steps: int = 120, use_precise: bool = False):
        """用本集合的真实统计跑**真实 DEM 铺粉**求解，返回 ``PowderBedResult``。

        惰性导入 ``amforge.powderbed``（→``diffmech``），使本模块可干净导入。
        DEM 在重力 + 刮刀约束下沉降，密实度 / 配位数 / 粗糙度为**解出的**，
        非假设。DEM 床的晶格规模由零件 bbox 与导入 d50 共同定出，保证物理代表。
        """
        from amforge.powderbed import solve_powderbed
        return solve_powderbed(
            geometry, process,
            params=self.powder_bed_params(
                n_steps=n_steps, geometry=geometry, use_precise=use_precise),
            powder_fidelity="dem")

    # ===================================================================
    # 6. GUI 切面叠加（取某 z 层附近的粒子 x,y）
    # ===================================================================
    def slice_xy(self, layer_z: float, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
        """返回距 ``layer_z`` 容差内的粒子 ``(xs, ys)``，供 2D 切片散点叠加。"""
        pos = self._pos_np()
        if self.dim == 2:
            return pos[:, 0], pos[:, 1]
        z = pos[:, 2]
        m = np.abs(z - float(layer_z)) <= float(tol)
        return pos[m, 0], pos[m, 1]

    def summary_str(self) -> str:
        s = self.stats()
        return (
            f"粒子数={s['n_particles']}，材料={s['material']} (ρ={s['rho']:.0f} kg/m³)\n"
            f"d50={s['d50']*1e6:.1f} µm，PSD σ={s['psd_sigma']:.2f}，"
            f"平均粒径={s['mean_diameter']*1e6:.1f} µm\n"
            f"整体体积分数={self.packing_density():.3f}"
        )


__all__ = ["ParticleCollection", "VALID_CSV_FORMATS"]
