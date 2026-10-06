"""高保真增材成形（塑性 / CPFE 生死单元 FEM）
===============================================

把 P2-② 的真实塑性/晶体塑性热-力求解器 :mod:`diffmech.methods.am.
thermomechanical_plastic` 接入 ForgeCore，产出与降级模型
:func:`amforge.buildup.solve_buildup` **同一契约** 的 ``AsBuiltPart``，
但残余应力/应变来自**逐层激活的弹塑性有限元 + 返回映射**，而不是趋势代理：

* 由体素 ``PartGeometry``（SDF）构建结构化 hex8 ``LayeredMesh``，覆盖整张栅格；
* 体素 ``ThermalHistory.peak_temperature`` 采样到 FEM 单元中心，换算成
  逐层热应变 ``ε_th = α_T·(T_peak − T_ref)``，并只在**零件内部**单元激活
  （element-birth 只在几何内发生）；
* 调用 ``solve_thermomechanical_plastic`` 做逐层弹塑性平衡 + J2/CPFE 返回映射，
  塑性状态跨层携带 → 残余应力被**封顶在屈服强度附近**（真实 LPBF 量级），
  塑性应变正确累积；
* 把 FEM 单元的应力/塑性应变/位移场**回填**到体素栅格，输出 ``AsBuiltPart``
  （Voigt 应力、Voigt 塑性应变、位移、变形后 SDF）。

整条链路对工艺参数（经 thermal→peak_T）可微：``jax.grad`` 可穿透。

注册名 ``asbuilt.thermomechanical_plastic``；用
``select={"asbuilt": "asbuilt.thermomechanical_plastic"}`` 显式启用，
不破坏默认降阶链路。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from amforge.core.contracts import (
    PartGeometry, ThermalHistory, AsBuiltPart,
)
from amforge.materials import get_material
from diffmech.methods.am.activation import build_layered_mesh, LayeredMesh
from diffmech.methods.am.thermomechanical_plastic import (
    solve_thermomechanical_plastic,
)
from diffmech.materials.plasticity import J2Plasticity
from diffmech.methods.cpfe.crystal_plasticity import CrystalPlasticity
from diffmech.methods.cpfe.slip_systems import (
    fcc_slip_systems, bcc_slip_systems,
)
from diffmech.solvers.boundary_conditions import DirichletBC

# 平滑张量 -> Voigt(3D) 组件顺序固定为 [sxx, syy, szz, sxy, syz, szx]
_VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]


def _tensor_to_voigt6(t: jnp.ndarray) -> jnp.ndarray:
    """(..., 3, 3) 对称张量 -> (..., 6) Voigt（工程剪应变 γ=2ε，故 *0.5 取平均）。"""
    comps = []
    for (a, b) in _VOIGT:
        if a == b:
            comps.append(t[..., a, b])
        else:
            comps.append(0.5 * (t[..., a, b] + t[..., b, a]))
    return jnp.stack(comps, axis=-1)


def _slip_systems_for(lattice: str | None):
    """按晶格选滑移系：bcc -> {110}<111>，其余 -> fcc {111}<110>。"""
    l = (lattice or "fcc").lower()
    if l == "bcc":
        return bcc_slip_systems()
    return fcc_slip_systems()


def _build_part_layered_mesh(geometry: PartGeometry, *, max_layers: int = 24):
    """由体素 PartGeometry 构建覆盖整张栅格的结构化 hex8 LayeredMesh。

    返回 ``(layered, nx, ny, nz)``，其中 ``nx = Nx-1, ny = Ny-1``，
    ``nz = n_layers*cells_per_layer ≤ Nz-1``（FEM 单元中心正好落在体素之间，
    便于后续 1:1 回填）。
    """
    shape = geometry.shape
    Nx, Ny, Nz = int(shape[0]), int(shape[1]), int(shape[2])
    spacing = float(geometry.spacing)
    lo, hi = geometry.bbox()
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)

    nx = max(Nx - 1, 1)
    ny = max(Ny - 1, 1)
    n_vox_z = max(Nz - 1, 1)
    cells_per_layer = max(1, n_vox_z // max(max_layers, 1))
    n_layers = max(1, n_vox_z // cells_per_layer)
    nz = n_layers * cells_per_layer

    # 构建方向 z 仅覆盖 nz 个单元（<= 整张栅格，避免越界）。
    z_min = float(lo[2])
    z_max = z_min + nz * spacing
    layer_zs = z_min + (np.arange(n_layers) + 0.5) * cells_per_layer * spacing
    bbox = [(float(lo[0]), float(hi[0])),
            (float(lo[1]), float(hi[1])),
            (z_min, z_max)]
    layered = build_layered_mesh(
        np.asarray(layer_zs), bbox, nx=nx, ny=ny,
        cells_per_layer=cells_per_layer,
    )
    return layered, nx, ny, nz, n_layers


def _sample_voxel_grid(grid: jnp.ndarray, cell_centers: np.ndarray,
                       origin: np.ndarray, spacing: float,
                       shape) -> jnp.ndarray:
    """把体素栅格按最近邻采样到 FEM 单元中心（gather，对 grid 可微）。"""
    Nx, Ny, Nz = int(shape[0]), int(shape[1]), int(shape[2])
    idx = jnp.clip(
        jnp.round((cell_centers - origin) / spacing),
        0, jnp.array([Nx - 1, Ny - 1, Nz - 1], dtype=jnp.float64),
    ).astype(jnp.int32)
    return grid[idx[:, 0], idx[:, 1], idx[:, 2]]


def _map_cells_to_voxels(field_cells: jnp.ndarray, geometry: PartGeometry,
                         nx: int, ny: int, nz: int) -> jnp.ndarray:
    """FEM 单元场 (n_cells, C) -> 体素场 (Nx, Ny, Nz, C)（最近邻回填，gather）。"""
    Nx, Ny, Nz = [int(s) for s in geometry.shape]
    f3 = field_cells.reshape(nx, ny, nz, -1)
    ix = jnp.clip(jnp.arange(Nx), 0, nx - 1)
    iy = jnp.clip(jnp.arange(Ny), 0, ny - 1)
    iz = jnp.clip(jnp.arange(Nz), 0, nz - 1)
    return f3[ix[:, None, None], iy[None, :, None], iz[None, None, :]]


def solve_asbuilt_plastic(*, geometry: PartGeometry, thermal: ThermalHistory,
                          params=None) -> AsBuiltPart:
    """高保真增材成形：塑性/CPFE 生死单元 FEM -> AsBuiltPart。

    注册名 ``asbuilt.thermomechanical_plastic``；consumes (geometry, thermal)；
    produces ``AsBuiltPart``。材料由 ``params['material']`` 解析（默认 316L），
    ``params['constitutive']`` 选 ``"j2"``（von Mises 各向同性硬化）或
    ``"cp"``（速率相关晶体塑性，按晶格选滑移系）。
    """
    p = dict(params or {})
    # 可微软占位覆盖：若提供（如来自体素 SDF 的 tanh 平滑 Heaviside），
    # 则用它替代硬占位做"激活掩码"与末态场回填，使成形对几何 SDF 可微——
    # 这是把 optimize_dimensional 升级为几何/形状（反变形）优化的关键使能项。
    occ_override = p.get("occupancy_override", None)

    if geometry.dim != 3:
        # 2D 几何走降阶模型（本高保真 FEM 面向 3D 增材构建）。
        from amforge.buildup import solve_buildup
        return solve_buildup(
            geometry=geometry,
            process=p.get("process"),
            thermal=thermal,
            microstructure=p.get("microstructure"),
            params=p,
        )

    mat = get_material(p.get("material", "316L"))
    constitutive = str(p.get("constitutive", "j2")).lower()

    # ---- 1. 几何 -> LayeredMesh ------------------------------------------
    (layered, nx, ny, nz, n_layers) = _build_part_layered_mesh(
        geometry, max_layers=int(p.get("max_layers", 24)))

    spacing = float(geometry.spacing)
    origin = np.asarray(geometry.origin, dtype=np.float64)
    shape = geometry.shape
    cell_centers = np.asarray(layered.cell_centers)  # (n_cells, 3) 静态

    # ---- 2. 体素场 -> FEM 单元 ------------------------------------------
    part_mask_cells = _sample_voxel_grid(
        occ_override if occ_override is not None else geometry.occupancy,
        cell_centers, origin, spacing, shape)  # (n_cells,)
    peak_T_cells = _sample_voxel_grid(
        thermal.peak_temperature, cell_centers, origin, spacing, shape)

    alpha_T = float(mat.alpha_thermal)
    T_ref = float(mat.T_ref_mech)
    dT_cells = alpha_T * (peak_T_cells - T_ref)  # (n_cells,) 热应变幅值

    layer_id = np.asarray(layered.layer_id)  # (n_cells,) 静态
    onehot = (layer_id[None, :] == np.arange(n_layers)[:, None]).astype(jnp.float64)
    # 仅本层沉积时该单元承受热载荷（逐层冷却收缩的简化）。
    thermal_strain_per_layer = dT_cells[None, :] * onehot  # (n_layers, n_cells)

    # ---- 3. 本构模型 ----------------------------------------------------
    if constitutive == "cp":
        dirs, norms = _slip_systems_for(mat.lattice)
        material = CrystalPlasticity(
            E=float(mat.E), nu=float(mat.nu),
            slip_directions=dirs, slip_normals=norms,
            gamma_dot0=float(p.get("gamma_dot0", 0.1)),
            m=float(mat.rate_exponent),
            g0=float(mat.crss0), h0=float(mat.cp_hardening),
            g_sat=float(mat.crss_sat),
        )
    else:
        material = J2Plasticity(
            E=float(mat.E), nu=float(mat.nu),
            sigma_y0=float(mat.sigma_y),
            H=float(mat.hardening_sat) * float(mat.hardening_rate),
        )

    # ---- 4. 基板约束（z=zmin 面全约束）--------------------------------
    nodes = np.asarray(layered.mesh.nodes)
    zmin = float(geometry.origin[2])
    bot = np.where(nodes[:, 2] < zmin + 0.5 * spacing)[0]
    dofs = (np.repeat(bot * 3, 3) + np.tile(np.arange(3), bot.shape[0])).astype(np.int64)
    plate_bc = DirichletBC.fixed(jnp.asarray(dofs), 0.0)

    # ---- 5. 逐层弹塑性求解 --------------------------------------------
    result = solve_thermomechanical_plastic(
        layered, material,
        thermal_strain_per_layer=thermal_strain_per_layer,
        dirichlet_bcs=[plate_bc],
        constitutive=constitutive,
        T_ref=T_ref, alpha_T=alpha_T,
        tau_activation=float(p.get("tau_activation", 1e-3)),
        dim=3,
        dt_cp=float(p.get("dt_cp", 1.0)),
        n_sub_cp=int(p.get("n_sub_cp", 256)),
        reg=1e-6,
        cell_mask=part_mask_cells,
    )

    # ---- 6. FEM 场回填到体素 ------------------------------------------
    residual_voigt = _tensor_to_voigt6(result.residual_stress)      # (n_cells,6)
    plastic_tensor = result.plastic_strain_history[-1]              # (n_cells,3,3)
    plastic_voigt = _tensor_to_voigt6(plastic_tensor)               # (n_cells,6)

    U = result.U_final
    nodes_disp = U.reshape(layered.mesh.n_nodes, 3)
    cell_disp = nodes_disp[layered.mesh.cells].mean(axis=1)         # (n_cells,3)

    vx_res = _map_cells_to_voxels(residual_voigt, geometry, nx, ny, nz)
    vx_pla = _map_cells_to_voxels(plastic_voigt, geometry, nx, ny, nz)
    vx_disp = _map_cells_to_voxels(cell_disp, geometry, nx, ny, nz)

    occ = occ_override if occ_override is not None else geometry.occupancy
    vx_res = vx_res * occ[..., None]
    vx_pla = vx_pla * occ[..., None]
    vx_disp = vx_disp * occ[..., None]

    # ---- 7. 变形后 SDF（沿法向推移）-----------------------------------
    grad = jnp.stack(jnp.gradient(geometry.sdf, spacing), axis=-1)
    glen = jnp.linalg.norm(grad, axis=-1, keepdims=True)
    n_hat = jax.lax.stop_gradient(grad / jnp.maximum(glen, 1e-12))
    sdf_def = geometry.sdf + jnp.sum(vx_disp * n_hat, axis=-1)

    return AsBuiltPart(
        sdf=jnp.asarray(sdf_def),
        displacement=jnp.asarray(vx_disp),
        residual_stress=jnp.asarray(vx_res),
        residual_strain=jnp.asarray(vx_pla),
        spacing=spacing,
        dim=3,
    )


__all__ = ["solve_asbuilt_plastic"]
