"""边界/初值条件（BC/IC）数据模型 + 体素几何选面 + 求解器应用

缺口 #20-②A：GUI 前处理「几何元素选面施 BC/IC」。

选面策略（体素 SDF 几何）
------------------------
以表面外法向 (``geometry.sdf_normal``) 的轴向分量把界面体素分到 6 个轴对齐面
（±X/±Y/±Z）+「全部表面」。这是体素几何下最稳健、物理最直观的选面方式：
``-Z`` 即基板/底面、``+Z`` 即自由顶面、``±X/±Y`` 即侧壁。对 2D 几何（dim==2）
不存在 Z 向面，``±Z`` 选择器自然返回空掩膜。

BC 类型（均作用于实体表面体素，平滑、对 sdf/几何可微）
------------------------------------------------------
* ``dirichlet`` : 固定温度 T_fixed（如基板恒温、对称面）——每步把该处焓强制覆盖；
* ``convection`` : 对流换热 ``h·(T_inf − T)``，单位与全局弱冷却一致，且在该面
  取代默认全局弱冷却（避免重复计入）；
* ``flux``       : 体积热流密度 ``q [W/m³]``（如辅助加热 / 强制冷却）；
* ``adiabatic``  : 绝热——移除该面的全局弱冷却项（视为热绝缘）。

IC 类型
------
* ``uniform`` / ``preheat`` : 全场初温 T0（如基板预热）。

所有 mask 均为平滑函数、对几何（sdf）、对工艺参数可微；Dirichlet 以
``jnp.where`` 覆盖，整条链路（含 ``jax.grad``）不被破坏。数据模型与求解器
解耦：GUI 前处理只产出 :class:`BoundaryCollection`，经 ``params["boundary_conditions"]``
可选喂入 ``solve_enthalpy_thermal``；缺省（None）时求解器保持原全局弱冷却行为。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from amforge import geometry as G
from amforge.core.contracts import PartGeometry, solid_mask
from amforge.thermal_enthalpy import enthalpy_of_temperature

# 轴向单位向量（按 dim 截断后取前 dim 分量）
_AXIS = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}
VALID_FACES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z", "all")
VALID_BC_KINDS = ("dirichlet", "convection", "flux", "adiabatic")
VALID_IC_KINDS = ("uniform", "preheat")


@dataclass(frozen=True)
class BoundaryCondition:
    """单条边界条件：选定几何面 + 类型 + 参数。

    Attributes
    ----------
    kind  : ``dirichlet`` / ``convection`` / ``flux`` / ``adiabatic``
    face  : 选面 ``±X/±Y/±Z/all``
    value : 主值（dirichlet=T_fixed[K]；convection=h[W/m²/K]；flux=q[W/m³]）
    value2: 次值（convection=T_inf[K]；其它未用）
    label : 可选中文标签
    """

    kind: str
    face: str
    value: float
    value2: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_BC_KINDS:
            raise ValueError(f"未知 BC 类型: {self.kind!r}（可选 {VALID_BC_KINDS}）")
        if self.face not in VALID_FACES:
            raise ValueError(f"未知选面: {self.face!r}（可选 {VALID_FACES}）")
        if self.kind == "convection" and self.value2 <= 0.0:
            raise ValueError("convection 需要 value2=T_inf(K) > 0")

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return {
            "dirichlet": f"Dirichlet {self.face} = {self.value:.0f} K",
            "convection": f"Convection {self.face} h={self.value:.2g}, T∞={self.value2:.0f} K",
            "flux": f"Flux {self.face} = {self.value:.2g} W/m³",
            "adiabatic": f"Adiabatic {self.face}",
        }[self.kind]


@dataclass(frozen=True)
class InitialCondition:
    """全场初值条件（目前支持均匀初温）。"""

    kind: str = "preheat"
    value: float = 373.0

    def __post_init__(self) -> None:
        if self.kind not in VALID_IC_KINDS:
            raise ValueError(f"未知 IC 类型: {self.kind!r}（可选 {VALID_IC_KINDS}）")


@dataclass(frozen=True)
class BoundaryCollection:
    """一组边界条件 + 初值条件，可序列化、可直接喂给求解器。"""

    bcs: tuple = ()
    ic: InitialCondition | None = None

    def with_bc(self, bc: BoundaryCondition) -> "BoundaryCollection":
        return BoundaryCollection(bcs=self.bcs + (bc,), ic=self.ic)

    def to_dict(self) -> dict:
        return {
            "bcs": [
                {"kind": b.kind, "face": b.face, "value": b.value,
                 "value2": b.value2, "label": b.label}
                for b in self.bcs
            ],
            "ic": None if self.ic is None
            else {"kind": self.ic.kind, "value": self.ic.value},
        }

    @staticmethod
    def from_dict(d: dict) -> "BoundaryCollection":
        bcs = tuple(BoundaryCondition(**b) for b in d.get("bcs", []))
        ic = InitialCondition(**d["ic"]) if d.get("ic") else None
        return BoundaryCollection(bcs=bcs, ic=ic)


# ---------------------------------------------------------------------------
# 选面：体素几何表面体素 → 轴向面掩膜
# ---------------------------------------------------------------------------
def face_mask(part: PartGeometry, face: str, *,
              angle_deg: float = 50.0, band_voxels: float = 2.0) -> jnp.ndarray:
    """返回选定面在体素栅格上的平滑掩膜（0/1，形状同 part.sdf）。

    以表面外法向 (``sdf_normal``) 与面轴向的夹角阈值把界面体素分到目标面；
    ``face="all"`` 取完整表面。掩膜对 sdf 可微（smoothstep 过渡），故可被
    ``jax.grad`` 穿透。2D 几何下 ``±Z`` 选择器返回 ≈0（不存在 Z 向面）。
    """
    if face not in VALID_FACES:
        raise ValueError(f"未知选面: {face!r}（可选 {VALID_FACES}）")
    n = G.sdf_normal(part)                       # (..., dim) 单位外法向
    sdf = part.sdf
    sp = float(part.spacing)
    band = jnp.exp(-(sdf / (band_voxels * sp)) ** 2)   # 界面带权重（平滑）
    surf = solid_mask(sdf).astype(jnp.float64) * band   # 实体表面体素
    if face == "all":
        return surf
    vec = jnp.asarray(_AXIS[face][: part.dim], dtype=jnp.float64)
    # 2D 几何不存在 Z 向面：轴索引越界 → 该面严格为空掩膜
    _AXIS_IDX = {"+X": 0, "-X": 0, "+Y": 1, "-Y": 1, "+Z": 2, "-Z": 2}
    if _AXIS_IDX[face] >= part.dim:
        return jnp.zeros_like(surf)
    cos_t = math.cos(math.radians(angle_deg))
    d = jnp.sum(n * vec, axis=-1)                # 法向在面轴向上的投影
    sel = 0.5 * (1.0 + jnp.tanh((d - cos_t) / 0.1))
    return surf * sel


# ---------------------------------------------------------------------------
# 求解器项：由 BC 集合生成 rhs 附加项 / 覆盖掩膜
# ---------------------------------------------------------------------------
def boundary_terms(part: PartGeometry, bcs: BoundaryCollection, *,
                   T: jnp.ndarray, dx, T_amb: float, hcool: float,
                   masks: dict | None = None):
    """由 BC 集合计算焓法 rhs 的附加项。

    返回 ``(extra_rhs, covered, dirichlet)``：
      * ``extra_rhs`` : 加进 rhs 的逐体素项（对流 + 热流）；
      * ``covered``   : 被显式 BC 覆盖的体素掩膜（这些处应移除全局弱冷却）；
      * ``dirichlet`` : ``(mask, T_fixed)`` 列表，供扫描每步把焓强制覆盖。

    全部为可微运算；``T`` 以 tracer 形式进入对流项，故对温度/几何可微。

    ``masks`` 为 ``{face: mask}`` 预计算表（求解器在时间扫描外算一次）：
    :func:`face_mask` 含全场 ``sdf_normal``，逐步重算会把显式推进的每步成本
    抬高一个量级。**量纲**：对流/热流是**表面**量 [W/m²]，除以 ``dx`` 摊到体素
    上才与 ``∂H/∂t`` [J/m³/s] 同量纲——此前直接相加，等价于把 h 当成了
    [W/m³/K]，网格越粗散热越被高估（A0 一并修正）。
    """
    extra = jnp.zeros_like(T)
    covered = jnp.zeros_like(T)
    dirichlet: list[tuple[jnp.ndarray, float]] = []
    for bc in bcs.bcs:
        m = face_mask(part, bc.face) if masks is None else masks[bc.face]
        if bc.kind == "dirichlet":
            dirichlet.append((m, float(bc.value)))
            covered = covered + m
        elif bc.kind == "convection":
            # h·(T_inf − T)/dx ；与全局 cool 同约定（rhs 中加 −h(T−T_inf)）
            extra = extra + m * float(bc.value) * (float(bc.value2) - T) / dx
            covered = covered + m
        elif bc.kind == "flux":
            extra = extra + m * float(bc.value)        # 体积热流密度 [W/m³]
            covered = covered + m
        elif bc.kind == "adiabatic":
            covered = covered + m                       # 仅移除全局冷却
    return extra, jnp.clip(covered, 0.0, 1.0), dirichlet


def initial_enthalpy_field(part: PartGeometry, bcs: BoundaryCollection, *,
                           rho, cp, L, T_amb, T_sol, T_liq) -> jnp.ndarray:
    """由 IC 生成初态焓场 H0；无 IC 时返回全 0（即 T_amb）。"""
    H0 = jnp.zeros_like(part.sdf)
    if bcs.ic is not None and bcs.ic.kind in ("uniform", "preheat"):
        H_pre = enthalpy_of_temperature(
            jnp.asarray(float(bcs.ic.value)), rho=rho, cp=cp, L=L,
            T_amb=T_amb, T_sol=T_sol, T_liq=T_liq)
        H0 = jnp.where(solid_mask(part.sdf) > 0.5, H_pre, H0)
    return H0


# ---------------------------------------------------------------------------
# 序列化辅助（供 GUI 导出 / 导入复用）
# ---------------------------------------------------------------------------
def export_bc(bc: BoundaryCollection, out_dir: str | Path) -> str:
    """把 BC/IC 集合写入 ``preproc_bc.json``，返回文件路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preproc_bc.json"
    path.write_text(json.dumps(bc.to_dict(), indent=2), encoding="utf-8")
    return str(path)


def import_bc(out_dir: str | Path) -> BoundaryCollection:
    """读回 :func:`export_bc` 导出的 BC/IC；文件不存在时返回空集合。"""
    path = Path(out_dir) / "preproc_bc.json"
    if not path.exists():
        return BoundaryCollection()
    return BoundaryCollection.from_dict(json.loads(path.read_text(encoding="utf-8")))
