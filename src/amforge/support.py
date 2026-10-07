"""支撑结构生成与仿真（Module A）—— SLM 悬垂/薄壁支撑的自动生成与热-力修正
================================================================================

商业软件（Simufact / ANSYS / 3DXpert）的标配能力，也是当前 STAGES 缺少
``support`` 阶段时 SLM 变形/失败预测的最大缺口。本模块把"支撑"显式建模为
一个与零件同栅格的占位/符号场 :class:`~amforge.core.contracts.SupportStructure`，
并给出两档求解器：

* :func:`support_auto` —— **代理默认档**（秒级，``cost=1``）：按悬垂角阈值规则
  自动判定需要支撑的区域（block 整足迹块支撑 / overhang 仅悬垂下方支撑），
  产出 :class:`SupportStructure`。这是绝大多数"先看看要多少支撑"场景的默认路径。
* :func:`support_simulate` —— **高保真档**（``cost=50``，``select`` 显式启用）：
  **复用** :func:`amforge.asbuilt_plastic.solve_asbuilt_plastic` 的逐层弹塑性
  生死单元 FEM 内核，把 ``geometry ∪ support`` 作为「合并几何」一起求解，
  再把结果按零件区域遮罩回 :class:`AsBuiltPart`——于是支撑以「额外约束 + 散热边界」
  的角色真实降低了零件变形/残余应力，而不是事后贴标签。

设计要点
--------
* 支撑不进默认轻量链路（``support.simulate`` cost=50，不会被 ``Pipeline.auto``
  自动选到；要用 ``select={'support':'support.simulate'}`` 显式切换）。
* ``geometry → support.auto → SupportStructure``；``SupportStructure`` 可作为
  额外输入喂给铺粉/热-力耦合（蓝图 DAG 边；本模块先把「支撑修正成形」这条边落地）。
* 全部为纯函数 + 契约通信，沿用 ``register_solver`` 统一签名，可无缝接入
  双注册表与 ``Pipeline``。

典型用法
--------
>>> from amforge.support import support_auto, support_simulate
>>> sup = support_auto(geometry=part, params={"kind": "block",
...                                           "overhang_angle_deg": 45.0})
>>> print(sup.summary())
>>> asbuilt = support_simulate(geometry=part, support=sup, process=plan,
...                            params={"material": "316L", "max_layers": 12})
"""

from __future__ import annotations

import jax.numpy as jnp

from amforge.core.contracts import (
    AsBuiltPart,
    PartGeometry,
    ProcessPlan,
    SupportStructure,
    solid_mask,
)
from amforge.core.registry import register_solver


# ---------------------------------------------------------------------------
# 辅助：栅格/列运算
# ---------------------------------------------------------------------------
def _part_occupancy(geometry: PartGeometry) -> jnp.ndarray:
    """硬占位（0/1），带 ulp 级容差（``sdf == 0`` 的刀锋集不随 dx 拼写翻面）。"""
    return solid_mask(geometry.sdf).astype(jnp.float64)


def _column_bottom(part: jnp.ndarray) -> jnp.ndarray:
    """每列中零件出现的最低 z 索引 (..., 1)；无零件列返回 nz（置于"无穷远"）。

    用于判定「该列零件是否落在基板上（bottom==0）」与「支撑需从 z=0 填到 zbot」。
    """
    nz = int(part.shape[-1])
    zs = jnp.arange(nz).reshape((1,) * (part.ndim - 1) + (nz,))
    return jnp.min(jnp.where(part > 0.5, zs, nz), axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 求解器 1：规则自动布点支撑（代理默认）
# ---------------------------------------------------------------------------
@register_solver(
    "support.auto",
    consumes=("PartGeometry",),
    produces="SupportStructure",
    stage="support",
    modality=("SLM",),
    differentiable=True,
    cost=1.0,
    doc="悬垂角阈值规则自动布点支撑（block / overhang）",
)
def support_auto(*, geometry: PartGeometry, params=None) -> SupportStructure:
    """按悬垂角阈值规则自动生成支撑结构。

    判定逻辑（沿构建方向 z 的逐列分析）：

    * **block（默认）**：零件下方、未与零件重叠、且零件底不在基板上的空域全部填支撑
      —— 即「整足迹块支撑」，最简单稳健，是离线快速预览的默认。
    * **overhang**：仅在出现「向下暴露面」（零件 voxel 其正下方为空）的列布支撑，
      即真正的悬垂区域；其余靠自身/邻列支撑的区域不布，更省料。

    参数
    ----
    params : dict
        ``kind``（"block"/"overhang"，默认 "block"）、
        ``overhang_angle_deg``（悬垂角阈值，默认 45°）、
        ``density``（支撑填充密度 0~1，默认 1.0，用于稀疏抽稀可视化与成本估算）。
    """
    p = dict(params or {})
    kind = str(p.get("kind", "block")).lower()
    overhang_angle_deg = float(p.get("overhang_angle_deg", 45.0))
    density = float(p.get("density", 1.0))
    spacing = float(geometry.spacing)

    sdf = geometry.sdf
    part = _part_occupancy(geometry)
    nz = int(sdf.shape[-1])
    zs = jnp.arange(nz).reshape((1,) * (sdf.ndim - 1) + (nz,))

    bottom = _column_bottom(part)                       # (..., 1)
    has_part = (bottom < nz)                             # 该列是否存在零件
    # 判据是物理的：**该列最低实体体素不在基板层（bottom==0）⇒ 零件与基板之间有空隙，
    # 需要支撑填充**。旧写法 `bottom >= 2` 是把"#19 前的严格 `sdf<0` 会把设计边界面
    # 那一整层剔掉"烤进了阈值——严格掩膜下底面正好落在体素中心上的零件 bottom 凭空 +1。
    # 2×2 实测（刀锋档夹具：底面=z0 的贴板件 / 离板 1 体素的悬空件，
    # docs/evidence/2026-10-07/am_t2_support_threshold.log）needs 列数：
    #   贴板件   (新掩膜,>=1)=0  (旧,>=1)=**4 假支撑**  (旧,>=2)=0  (新,>=2)=0
    #   悬空件   (新掩膜,>=1)=4  (旧,>=1)=4             (旧,>=2)=4  (新,>=2)=**0 漏判**
    # ⇒ 掩膜与阈值是**一对**：正确的两种自洽组合是 (旧掩膜,>=2) 与 (新掩膜,>=1)，
    #   交叉组合一个造假支撑、一个漏判 1 体素间隙的悬空件。**回退必须同批**，
    #   单独改任一侧都会破坏 tests/test_support.py 的这两条。
    # 出厂组合实测：贴板件 volume_fraction=0.000000/contact_area=0，悬空件 0.018519。
    needs = has_part & (bottom >= 1)
    # 在「含零件且有间隙」的列中，填充零件底面之下、且与零件不重叠的空域
    under = needs & (zs < bottom) & (part < 0.5)

    if kind == "overhang":
        below = jnp.concatenate(
            [jnp.zeros((*part.shape[:2], 1)), part[..., :-1]], axis=-1)
        down_exposed = (part > 0.5) & (below < 0.5)
        col_overhang = jnp.any(down_exposed, axis=-1, keepdims=True)
        support = under & col_overhang
    else:
        support = under

    # 稀疏抽稀（建模低密度支撑/成本估算）：按 (x+y+z) 奇偶保留约一半，
    # 但始终保留贴板首层以保证支撑与基板连通。density>=0.5 不抽稀。
    if density < 1.0 - 1e-6:
        coords = jnp.meshgrid(
            jnp.arange(sdf.shape[0]), jnp.arange(sdf.shape[1]),
            jnp.arange(nz), indexing="ij")
        keep = (((coords[0] + coords[1] + coords[2]) % 2 == 0)
                | (zs < 1))
        support = support & keep

    support = support.astype(jnp.float64)
    support_mask = support
    # 支撑 SDF：负在支撑内部（与 PartGeometry.sdf 同栅格、同轴，
    # 便于用 min(part_sdf, support_sdf) 求「零件 ∪ 支撑」）。
    support_sdf = (0.5 - support_mask) * spacing

    # 接触面积：支撑 voxel 其 +z 邻居为零件（即支撑顶到零件底面）。
    up = jnp.concatenate(
        [part[..., 1:], jnp.zeros((*part.shape[:2], 1))], axis=-1)
    contact = support * (up > 0.5)
    cell_area = spacing ** 2
    contact_area = jnp.sum(contact) * cell_area
    volume_fraction = jnp.sum(support) / jnp.prod(jnp.asarray(sdf.shape))

    return SupportStructure(
        support_sdf=jnp.asarray(support_sdf),
        support_mask=jnp.asarray(support_mask),
        kind=kind,
        contact_area=jnp.asarray(contact_area, dtype=jnp.float64),
        volume_fraction=jnp.asarray(volume_fraction, dtype=jnp.float64),
        params={
            "kind": kind,
            "overhang_angle_deg": overhang_angle_deg,
            "density": density,
            "spacing": spacing,
        },
    )


# ---------------------------------------------------------------------------
# 求解器 2：高保真支撑-零件耦合成形（复用 asbuilt 热-力内核）
# ---------------------------------------------------------------------------
@register_solver(
    "support.simulate",
    consumes=("PartGeometry", "SupportStructure", "ProcessPlan"),
    produces="AsBuiltPart",
    stage="support",
    modality=("SLM",),
    differentiable=True,
    cost=50.0,
    doc="合并几何(零件∪支撑)复用 asbuilt 逐层热-力耦合，支撑作为约束/散热边界修正成形",
)
def support_simulate(*, geometry: PartGeometry, support: SupportStructure,
                     process: ProcessPlan, params=None) -> AsBuiltPart:
    """把支撑作为「额外约束 + 散热边界」并入逐层热-力耦合，修正成形件。

    真实复用 :func:`amforge.asbuilt_plastic.solve_asbuilt_plastic`：

    1. 由 ``min(geometry.sdf, support.support_sdf)`` 得到「零件 ∪ 支撑」合并几何；
    2. 取热历史（外部注入 ``params['thermal']``，否则在合并几何上跑
       meltpool + thermal 真实链）；
    3. 合并几何整体做逐层弹塑性求解——支撑落于基板，因此把零件「撑住」，
       残余应力被封顶、变形被抑制；
    4. 结果按**零件区域**遮罩回 :class:`AsBuiltPart`（支撑只作为边界条件参与，
       不出现在最终零件里）。

    这与蓝图「SupportStructure + AsBuiltPart → support.simulate 修正成形」的 DAG 边
    一致，且是**实打实**的 FEM 耦合，而非事后缩放。

    参数
    ----
    params : dict
        ``thermal``（可选注入的 :class:`ThermalHistory`）、``material``、
        ``max_layers``、``constitutive`` 等透传给 asbuilt 塑性求解器。
    """
    p = dict(params or {})

    # 合并几何：零件 ∪ 支撑
    union_sdf = jnp.minimum(geometry.sdf, support.support_sdf)
    union = PartGeometry(
        sdf=union_sdf,
        origin=geometry.origin,
        spacing=geometry.spacing,
        dim=geometry.dim,
        name=(geometry.name or "part") + "+support",
    )

    thermal = p.get("thermal", None)
    if thermal is None:
        from amforge.meltpool import solve_meltpool_surrogate
        from amforge.thermal import solve_thermal_history
        mp = solve_meltpool_surrogate(
            geometry=union, process=process,
            params={"n_grid": int(p.get("n_grid", 16))})
        thermal = solve_thermal_history(
            geometry=union, process=process, meltpool=mp, params={})

    # 透传 asbuilt 塑性求解器关心的参数（避免把 thermal 等无关键透传进去）
    _fwd = ("material", "max_layers", "constitutive", "gamma_dot0",
            "dt_cp", "n_sub_cp", "tau_activation")
    ab_params = {k: p[k] for k in _fwd if k in p}

    from amforge.asbuilt_plastic import solve_asbuilt_plastic
    ab_combined = solve_asbuilt_plastic(geometry=union, thermal=thermal,
                                        params=ab_params)

    # 只保留零件区域（支撑仅作为约束/散热边界参与求解）
    part_occ = _part_occupancy(geometry)  # (...,)
    return AsBuiltPart(
        sdf=geometry.sdf,
        displacement=ab_combined.displacement * part_occ[..., None],
        residual_stress=ab_combined.residual_stress * part_occ[..., None],
        residual_strain=ab_combined.residual_strain * part_occ[..., None],
        spacing=geometry.spacing,
        dim=geometry.dim,
    )


__all__ = ["support_auto", "support_simulate"]
