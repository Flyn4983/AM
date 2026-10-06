"""几何模块 —— 任意复杂构型的统一入口
=====================================

功能 1 要求"实现**任意复杂几何构型**的增材制造模拟"。本模块负责把任何几何
来源统一成 :class:`~amforge.core.contracts.PartGeometry`（体素化 SDF）：

============================  ==================================================
几何来源                      入口
============================  ==================================================
CAD 导出的 STL/OBJ 网格       :func:`from_stl` / :func:`from_mesh`
解析式基元 + 布尔运算 (CSG)   :func:`from_sdf_fn` + :mod:`primitives`
体素/CT 重建的 0-1 占位场     :func:`from_occupancy`
点阵/晶格 (AM 杀手级应用)     :func:`gyroid` / :func:`schwarz_p` / :func:`bcc_lattice`
拓扑优化的连续密度场          :func:`from_density`
============================  ==================================================

为什么统一到体素 SDF 而不是保留 CAD 网格？
------------------------------------------
1. **可微**：SDF 是标量场，几何本身可作为优化变量；网格的连通性变化不可微。
2. **单一栅格贯穿全链**：分层切片、扫描路径、生死单元激活、几何偏差比较
   全在同一栅格上做，避免跨模块网格映射带来的插值误差——这是多物理场
   耦合软件里最常见的精度杀手。
3. **拓扑无关**：内部孔洞、薄壁、点阵、多连通体一视同仁，不需要网格修补。

代价是分辨率受内存限制。工程上的做法是**分区多分辨率**：
零件级用粗栅格（~层厚量级），熔池级用局部细栅格
（见 :mod:`amforge.meltpool` 的 ``local_refine``）。

底层复用
--------
SDF 基元、平滑布尔、网格→SDF（最近点距离 + 广义绕数定号）复用
:mod:`diffmech.methods.am.geometry`；本模块只做契约化封装、
体素化采样与 AM 专有的几何诊断（悬垂、薄壁、可打印性）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from amforge.core.contracts import PartGeometry
from amforge.core.registry import register_solver

# --- 复用 diffmech 的 SDF 工具箱 -------------------------------------------
from diffmech.methods.am.geometry import (  # noqa: F401  (转出为公开 API)
    sdf_box,
    sdf_cylinder,
    sdf_gear,
    sdf_sphere,
    sdf_torus,
    smooth_difference,
    smooth_intersection,
    smooth_union,
)
from diffmech.methods.am.geometry import (
    difference,
    intersection,
    rotate_z,
    sdf_from_mesh,
    translate,
    union,
)

__all__ = [
    # 构造
    "from_sdf_fn",
    "from_mesh",
    "from_stl",
    "from_occupancy",
    "from_density",
    # 基元与布尔（转出）
    "sdf_sphere",
    "sdf_box",
    "sdf_cylinder",
    "sdf_torus",
    "sdf_gear",
    "union",
    "intersection",
    "difference",
    "translate",
    "rotate_z",
    "smooth_union",
    "smooth_intersection",
    "smooth_difference",
    # AM 专用几何
    "gyroid",
    "schwarz_p",
    "bcc_lattice",
    "with_baseplate",
    # 采样与切片
    "sample_sdf",
    "slice_masks",
    "layer_z_heights",
    "resample",
    "crop_to_part",
    # 诊断
    "sdf_normal",
    "overhang_field",
    "overhang_fraction",
    "thin_wall_field",
    "printability_report",
    "read_stl",
]


# ===========================================================================
# 1. 构造：各种来源 -> PartGeometry
# ===========================================================================
def _grid_axes(bounds: Sequence[tuple[float, float]], spacing: float):
    """由包围盒与体素边长生成等距轴坐标。"""
    axes = []
    for lo, hi in bounds:
        n = max(2, int(np.ceil((hi - lo) / spacing)) + 1)
        axes.append(lo + spacing * np.arange(n, dtype=np.float64))
    return axes


def from_sdf_fn(
    sdf_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    bounds: Sequence[tuple[float, float]],
    spacing: float,
    name: str = "part",
    chunk: int | None = 200_000,
) -> PartGeometry:
    """把解析/隐式 SDF 函数体素化成 :class:`PartGeometry`。

    Parameters
    ----------
    sdf_fn
        ``f(x: (..., 3)) -> (...,)``，负值在内部，单位 m。
    bounds
        ``[(x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)]`` [m]。
        建议留 2~3 个体素的余量，避免零件贴边导致法向估计失真。
    spacing
        体素边长 [m]。经验取值：零件级 ≈ 层厚（SLM 30~50 µm），
        熔池级 ≈ 光斑半径 / 8。
    chunk
        分块求值的点数上限，控制峰值显存。``None`` 表示一次算完。

    Notes
    -----
    体素化本身是**预处理**，用 numpy/JAX 混合求值；得到的 ``sdf`` 数组
    进入契约后就是可微叶子，后续对它求梯度即"形状导数"。
    """
    axes = _grid_axes(bounds, spacing)
    grids = np.meshgrid(*axes, indexing="ij")
    pts = np.stack(grids, axis=-1).reshape(-1, 3)

    if chunk is None or pts.shape[0] <= chunk:
        vals = np.asarray(sdf_fn(jnp.asarray(pts)))
    else:
        out = []
        for i in range(0, pts.shape[0], chunk):
            out.append(np.asarray(sdf_fn(jnp.asarray(pts[i : i + chunk]))))
        vals = np.concatenate(out)

    sdf = vals.reshape([len(a) for a in axes])
    origin = np.asarray([a[0] for a in axes], dtype=np.float64)
    return PartGeometry(
        sdf=jnp.asarray(sdf, dtype=jnp.float64),
        origin=jnp.asarray(origin),
        spacing=float(spacing),
        dim=3,
        name=name,
    )


def read_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取二进制/ASCII STL，返回 ``(verts, tris)``，**不依赖 meshio**。

    STL 是 AM 的事实标准交换格式。这里直接解析，避免为了读一个三角网格
    而引入额外依赖——工程软件的依赖越少，部署到客户机器上越省事。
    """
    from diffmech.methods.am.particle_am import (
        _parse_ascii_stl,
        parse_binary_stl,
    )

    data = Path(path).read_bytes()
    # ASCII STL 以 "solid" 开头；但某些二进制文件的 80 字节头也可能以 solid 开头，
    # 因此用长度一致性做二次判定：二进制 STL 长度 == 84 + 50 * n_tri。
    is_ascii = data[:5].lower() == b"solid"
    if is_ascii and len(data) >= 84:
        n_tri = int(np.frombuffer(data[80:84], dtype="<u4")[0])
        if len(data) == 84 + 50 * n_tri:
            is_ascii = False
    if is_ascii:
        verts, tris = _parse_ascii_stl(data.decode("utf-8", errors="replace"))
    else:
        verts, tris = parse_binary_stl(data)
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def from_mesh(
    verts,
    tris,
    *,
    spacing: float,
    pad: float | None = None,
    name: str = "part",
    winding_order: int = 1,
    scale: float = 1.0,
    chunk: int = 50_000,
) -> PartGeometry:
    """任意封闭三角网格 -> 体素 SDF。

    符号用广义绕数（solid angle）判定，对非凸、多连通、薄壁体都稳健；
    距离用最近点距离，精度到亚体素。

    Parameters
    ----------
    scale
        单位换算系数。STL 常以 mm 为单位导出，此时传 ``scale=1e-3`` 转成 m。
    pad
        包围盒外扩量 [m]，默认 3 个体素。
    """
    verts = np.asarray(verts, dtype=np.float64) * float(scale)
    tris = np.asarray(tris, dtype=np.int64)
    if pad is None:
        pad = 3.0 * spacing

    lo = verts.min(axis=0) - pad
    hi = verts.max(axis=0) + pad
    bounds = [(float(lo[d]), float(hi[d])) for d in range(3)]

    fn = sdf_from_mesh(
        jnp.asarray(verts), jnp.asarray(tris),
        winding_order=winding_order, chunk=chunk,
    )
    return from_sdf_fn(fn, bounds=bounds, spacing=spacing, name=name, chunk=chunk)


def from_stl(
    path: str | Path,
    *,
    spacing: float,
    scale: float = 1.0,
    pad: float | None = None,
    name: str | None = None,
    winding_order: int = 1,
    chunk: int = 50_000,
) -> PartGeometry:
    """CAD 导出的 STL -> :class:`PartGeometry`（工程上最常用的入口）。

    Examples
    --------
    >>> part = from_stl("bracket.stl", spacing=50e-6, scale=1e-3)  # doctest: +SKIP
    >>> part.shape, float(part.volume())                           # doctest: +SKIP
    """
    verts, tris = read_stl(path)
    return from_mesh(
        verts, tris, spacing=spacing, pad=pad,
        name=name or Path(path).stem, winding_order=winding_order,
        scale=scale, chunk=chunk,
    )


def from_occupancy(
    occupancy,
    *,
    spacing: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    name: str = "part",
) -> PartGeometry:
    """0-1 占位体素场 -> 带符号距离场。

    用于 CT 扫描重建、体素建模器输出、或已有的 as-built 实测数据。
    内部用欧氏距离变换（EDT）求内外距离并合成 SDF：
    ``sdf = d_out - d_in``，界面处为 0。
    """
    occ = np.asarray(occupancy) > 0.5
    try:
        from scipy.ndimage import distance_transform_edt as edt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "from_occupancy 需要 scipy（用于欧氏距离变换）。"
            "请 pip install scipy，或改用 from_sdf_fn 直接给出 SDF。"
        ) from exc

    d_out = edt(~occ) * spacing
    d_in = edt(occ) * spacing
    sdf = np.where(occ, -d_in, d_out)
    # EDT 给的是体素中心到最近异类体素中心的距离，界面被系统性外推半个体素，
    # 减去 0.5*spacing 把零等值面挪回真实界面位置。
    sdf = sdf - 0.5 * spacing * np.sign(sdf)
    return PartGeometry(
        sdf=jnp.asarray(sdf, dtype=jnp.float64),
        origin=jnp.asarray(origin, dtype=jnp.float64),
        spacing=float(spacing),
        dim=int(occ.ndim),
        name=name,
    )


def from_density(
    density,
    *,
    spacing: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    threshold: float = 0.5,
    width: float | None = None,
    name: str = "topopt",
) -> PartGeometry:
    """拓扑优化的连续密度场 ρ∈[0,1] -> 可微 SDF。

    与 :func:`from_occupancy` 不同，这里**保持可微**：
    用 ``sdf ≈ width * (threshold - ρ)`` 的线性映射，
    梯度可以从 AM 仿真一路回传到密度场，实现
    "拓扑优化 + 可制造性" 的联合设计（功能 2 的自然延伸）。
    """
    rho = jnp.asarray(density, dtype=jnp.float64)
    if width is None:
        width = 2.0 * spacing
    sdf = width * (threshold - rho)
    return PartGeometry(
        sdf=sdf,
        origin=jnp.asarray(origin, dtype=jnp.float64),
        spacing=float(spacing),
        dim=int(rho.ndim),
        name=name,
    )


# ===========================================================================
# 2. AM 专用几何：点阵与基板
# ===========================================================================
def gyroid(*, cell: float, thickness: float = 0.0) -> Callable:
    """Gyroid 三周期极小曲面（TPMS）点阵的 SDF。

    增材制造最典型的"传统工艺做不出来"的结构：轻量化夹层、骨植入体、
    换热器芯体都在用。近似 SDF：``|sin·cos 三项和| - t``。

    Parameters
    ----------
    cell
        单胞尺寸 [m]。
    thickness
        壁厚参数（>0 得到片状 gyroid，=0 得到曲面本身）。
    """
    k = 2.0 * np.pi / cell

    def fn(x):
        X, Y, Z = k * x[..., 0], k * x[..., 1], k * x[..., 2]
        f = (jnp.sin(X) * jnp.cos(Y) + jnp.sin(Y) * jnp.cos(Z)
             + jnp.sin(Z) * jnp.cos(X))
        # 除以 |∇f| 的量级把隐式函数近似归一成距离（一阶近似）
        return (jnp.abs(f) - thickness * k) / k

    return fn


def schwarz_p(*, cell: float, thickness: float = 0.0) -> Callable:
    """Schwarz Primitive TPMS 点阵 SDF（比 gyroid 各向异性更强）。"""
    k = 2.0 * np.pi / cell

    def fn(x):
        f = (jnp.cos(k * x[..., 0]) + jnp.cos(k * x[..., 1])
             + jnp.cos(k * x[..., 2]))
        return (jnp.abs(f) - thickness * k) / k

    return fn


def bcc_lattice(*, cell: float, strut_radius: float) -> Callable:
    """体心立方杆系点阵（strut-based lattice）的近似 SDF。

    以单胞内 4 条体对角线为轴的圆柱并集。相比 TPMS，杆系点阵更容易
    对标经典桁架理论，也更常见于工业轻量化件。
    """
    dirs = np.array(
        [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]], dtype=np.float64
    ) / np.sqrt(3.0)
    dirs = jnp.asarray(dirs)

    def fn(x):
        # 折叠到单胞并平移到中心
        p = jnp.mod(x, cell) - 0.5 * cell
        # 点到过原点、方向 d 的直线距离
        d2 = []
        for i in range(4):
            d = dirs[i]
            proj = jnp.sum(p * d, axis=-1, keepdims=True) * d
            d2.append(jnp.linalg.norm(p - proj, axis=-1))
        return jnp.min(jnp.stack(d2, axis=0), axis=0) - strut_radius

    return fn


def with_baseplate(
    part: PartGeometry,
    *,
    thickness: float,
    name_suffix: str = "+plate",
) -> PartGeometry:
    """在零件下方加一层基板（substrate），栅格沿 Z 向下扩展。

    基板是 AM 热学的关键边界：它是主要散热通道，也是残余应力与
    变形的约束来源。不含基板的热分析会显著高估温度、低估残余应力。
    """
    n_add = max(1, int(np.ceil(thickness / part.spacing)))
    nx, ny, _nz = part.shape
    # 基板视为完全实体：SDF 取负的、随深度增大的距离
    depth = part.spacing * (np.arange(n_add, 0, -1, dtype=np.float64))
    plate = -np.broadcast_to(depth, (nx, ny, n_add)).copy()
    sdf = jnp.concatenate([jnp.asarray(plate), part.sdf], axis=-1)
    origin = part.origin.at[-1].add(-part.spacing * n_add)
    return PartGeometry(
        sdf=sdf, origin=origin, spacing=part.spacing,
        dim=part.dim, name=part.name + name_suffix,
    )


# ===========================================================================
# 3. 采样、切片、重采样
# ===========================================================================
def sample_sdf(part: PartGeometry, points) -> jnp.ndarray:
    """在任意物理坐标处三线性插值体素 SDF（可微，对点坐标也可微）。

    这是跨模块几何查询的唯一通道：粒子法要问"我在零件里吗"、
    熔池局部细网格要问"边界在哪"，都走这里，保证几何一致性。
    """
    pts = jnp.asarray(points, dtype=part.sdf.dtype)
    idx = (pts - part.origin) / part.spacing            # 物理坐标 -> 体素索引
    coords = [idx[..., d] for d in range(part.dim)]
    return jax.scipy.ndimage.map_coordinates(
        part.sdf, coords, order=1, mode="nearest"
    )


def layer_z_heights(part: PartGeometry, layer_thickness: float) -> np.ndarray:
    """逐层构建高度（沿 Z 轴）[m]。"""
    lo = float(np.asarray(part.origin)[-1])
    hi = lo + part.spacing * (part.shape[-1] - 1)
    n = max(1, int(np.ceil((hi - lo) / layer_thickness)))
    # 取每层中面高度，避免正好落在层界上导致占位判断抖动
    return lo + layer_thickness * (np.arange(n, dtype=np.float64) + 0.5)


def slice_masks(
    part: PartGeometry,
    layer_thickness: float,
    *,
    smoothness: float | None = None,
) -> tuple[jnp.ndarray, np.ndarray]:
    """分层切片：返回 ``(masks, z_heights)``。

    ``masks`` 形状 ``(n_layers, nx, ny)``，取值 [0,1] 的**可微**占位度。
    平滑 Heaviside 的宽度 ``smoothness`` 默认取 1 个体素——
    太小会让梯度稀疏（只有界面体素有梯度），太大会让几何失真。

    切片是"几何 → 工艺"的桥梁：扫描路径在每层 mask 上生成。
    """
    if smoothness is None:
        smoothness = float(part.spacing)
    zs = layer_z_heights(part, layer_thickness)

    nx, ny = part.shape[0], part.shape[1]
    ix, iy = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    z0 = part.origin[-1]

    def one(z):
        kz = jnp.full_like(ix, 0.0, dtype=part.sdf.dtype) + (z - z0) / part.spacing
        s = jax.scipy.ndimage.map_coordinates(
            part.sdf, [ix.astype(part.sdf.dtype), iy.astype(part.sdf.dtype), kz],
            order=1, mode="nearest",
        )
        return 0.5 * (1.0 - jnp.tanh(s / smoothness))

    masks = jax.vmap(one)(jnp.asarray(zs))
    return masks, zs


def resample(part: PartGeometry, spacing: float, *, name: str | None = None) -> PartGeometry:
    """改变体素分辨率（三线性插值）。

    典型用法：零件级用 100 µm 粗栅格跑全局热分析，
    局部热点区域 :func:`crop_to_part` 后重采样到 10 µm 跑熔池 CFD。
    """
    lo = np.asarray(part.origin, dtype=np.float64)
    hi = lo + part.spacing * (np.asarray(part.shape) - 1.0)
    bounds = [(float(lo[d]), float(hi[d])) for d in range(part.dim)]
    axes = _grid_axes(bounds, spacing)
    grids = jnp.meshgrid(*[jnp.asarray(a) for a in axes], indexing="ij")
    pts = jnp.stack(grids, axis=-1)
    sdf = sample_sdf(part, pts)
    return PartGeometry(
        sdf=sdf,
        origin=jnp.asarray([a[0] for a in axes]),
        spacing=float(spacing),
        dim=part.dim,
        name=name or part.name,
    )


def crop_to_part(part: PartGeometry, *, margin: float = 0.0) -> PartGeometry:
    """裁剪到零件紧包围盒 + margin，去掉大片空气体素。

    在 AM 里这一步很值：细长悬臂件的包围盒常有 80% 是空气，
    裁掉后热分析的自由度直接降一个量级。
    """
    occ = np.asarray(part.occupancy) > 0.5
    if not occ.any():
        return part
    idx = [np.where(occ.any(axis=tuple(j for j in range(part.dim) if j != d)))[0]
           for d in range(part.dim)]
    pad = int(np.ceil(margin / part.spacing))
    sl, new_origin = [], []
    for d in range(part.dim):
        a = max(0, int(idx[d][0]) - pad)
        b = min(part.shape[d], int(idx[d][-1]) + 1 + pad)
        sl.append(slice(a, b))
        new_origin.append(float(np.asarray(part.origin)[d]) + a * part.spacing)
    return PartGeometry(
        sdf=part.sdf[tuple(sl)],
        origin=jnp.asarray(new_origin),
        spacing=part.spacing,
        dim=part.dim,
        name=part.name,
    )


# ===========================================================================
# 4. 可打印性诊断
# ===========================================================================
def sdf_normal(part: PartGeometry) -> jnp.ndarray:
    """SDF 梯度（≈外法向），形状 ``(..., dim)``，已归一化。"""
    # 注意：不要传 edge_order —— JAX 的 jnp.gradient 不支持该参数
    # （numpy 支持；JAX 0.4+ 只实现了 edge_order=1 的等价行为）。
    grads = jnp.stack(jnp.gradient(part.sdf, part.spacing), axis=-1)
    norm = jnp.linalg.norm(grads, axis=-1, keepdims=True)
    return grads / jnp.maximum(norm, 1e-12)


def overhang_field(part: PartGeometry, *, angle_threshold: float = 45.0) -> jnp.ndarray:
    """悬垂严重度场 [0,1]：朝下表面与水平面夹角小于阈值处为 1。

    悬垂是 SLM 最主要的缺陷源（下表面粗糙、挂渣、需要支撑）。
    判据：表面法向 Z 分量 ``n_z < -cos(θ)`` 即认为是需支撑的悬垂面。
    这里给**平滑**版本，可直接作为可制造性惩罚项进损失函数。
    """
    n = sdf_normal(part)
    nz = n[..., -1]
    c = np.cos(np.deg2rad(angle_threshold))
    # 只在界面附近（|sdf| < 1.5 体素）计入
    band = jnp.exp(-(part.sdf / (1.5 * part.spacing)) ** 2)
    return band * 0.5 * (1.0 - jnp.tanh((nz + c) / 0.1))


def overhang_fraction(part: PartGeometry, *, angle_threshold: float = 45.0) -> jnp.ndarray:
    """悬垂面积占比（可微标量），可直接进损失函数做可制造性约束。"""
    of = overhang_field(part, angle_threshold=angle_threshold)
    band = jnp.exp(-(part.sdf / (1.5 * part.spacing)) ** 2)
    return jnp.sum(of) / jnp.maximum(jnp.sum(band), 1e-12)


def thin_wall_field(part: PartGeometry, *,
                    min_thickness: float | None = None) -> jnp.ndarray:
    """薄壁风险场：内部点到最近表面距离小于 ``min_thickness/2`` 处为 1。

    薄壁在 SLM 中要么打不出来，要么因散热不足而过熔。
    ``-sdf`` 正好是内部点到表面的距离，判据非常直接。

    ``min_thickness`` 缺省时取 3 个体素 —— 体素尺寸通常按层厚/光斑选取，
    所以这个缺省值本身就带着"3 倍特征尺度以下算薄壁"的工程含义。
    """
    if min_thickness is None:
        min_thickness = 3.0 * float(part.spacing)
    depth = jnp.maximum(-part.sdf, 0.0)
    return part.soft_occupancy() * 0.5 * (
        1.0 - jnp.tanh((depth - 0.5 * min_thickness) / (0.5 * part.spacing))
    )


def printability_report(
    part: PartGeometry,
    *,
    layer_thickness: float = 40e-6,
    angle_threshold: float = 45.0,
    min_thickness: float | None = None,
) -> dict[str, float]:
    """一份可打印性速查表，用于开工前的几何体检。

    返回体积、层数、悬垂占比、薄壁占比、体素数等。
    纯诊断，不影响仿真；但能在花 3 小时算熔池之前发现"这零件根本没法打"。
    """
    if min_thickness is None:
        min_thickness = 3.0 * layer_thickness
    n_layers = part.layer_count(layer_thickness)
    return {
        "voxels": int(np.prod(part.shape)),
        "shape": tuple(part.shape),
        "spacing_um": float(part.spacing * 1e6),
        "volume_mm3": float(part.volume()) * 1e9,
        "n_layers": int(n_layers),
        "overhang_fraction": float(overhang_fraction(part, angle_threshold=angle_threshold)),
        "thin_wall_fraction": float(
            jnp.sum(thin_wall_field(part, min_thickness=min_thickness))
            / jnp.maximum(jnp.sum(part.soft_occupancy()), 1e-12)
        ),
        "min_thickness_um": float(min_thickness * 1e6),
    }


# ===========================================================================
# 5. 注册为求解器（让 Pipeline 能从"无输入"开始生成几何）
# ===========================================================================
@register_solver(
    "geometry.parametric",
    consumes=(),
    produces="PartGeometry",
    stage="geometry",
    cost=1.0,
    differentiable=True,
    defaults={"sdf_fn": None, "bounds": None, "spacing": 50e-6, "name": "part"},
    doc="由解析 SDF 函数体素化生成几何（供形状/拓扑优化闭环使用）",
)
def solve_parametric_geometry(*, params) -> PartGeometry:
    """求解器封装：``params`` 需给出 ``sdf_fn`` 与 ``bounds``。

    之所以把"生成几何"也做成求解器，是为了让形状优化闭环
    （几何参数 → 工艺 → 仿真 → 损失）能整条挂在同一个 Pipeline 上。
    """
    p = dict(params or {})
    fn = p.get("sdf_fn")
    if fn is None:
        raise ValueError(
            "geometry.parametric 需要 params['sdf_fn']（一个 SDF 可调用对象）。"
            "如果你已经有 PartGeometry，直接把它作为 pipeline 的 given 输入即可。"
        )
    bounds = p.get("bounds")
    if bounds is None:
        raise ValueError("geometry.parametric 需要 params['bounds'] = [(lo,hi)]*3")
    return from_sdf_fn(
        fn, bounds=bounds, spacing=float(p.get("spacing", 50e-6)),
        name=str(p.get("name", "part")),
    )
