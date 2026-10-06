"""Minimal end-to-end smoke for the true-3D layered AM FEM engine in diffmech.

Proves that an arbitrary part geometry (here a box, but any SDF/mesh works)
can be simulated layer-by-layer with thermo-mechanical coupling: a structured
hex8 LayeredMesh is built, the build-plate is clamped, and per-layer cooling
thermal strain drives a differentiable residual-stress / distortion solve.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from diffmech.materials import LinearElasticIsotropic
from diffmech.solvers import DirichletBC
from diffmech.methods.am.activation import build_layered_mesh
from diffmech.methods.am.thermomechanical import solve_thermomechanical

# --- 1) arbitrary geometry: a box part bounding box (any SDF/mesh → bbox) ----
bbox = [(-1.0e-3, 1.0e-3), (-1.0e-3, 1.0e-3), (0.0, 2.0e-3)]  # 2mm cube
nx = ny = 8
n_layers = 4
cells_per_layer = 1
layer_zs = np.linspace(bbox[2][0], bbox[2][1], n_layers, endpoint=False) + \
    (bbox[2][1] - bbox[2][0]) / (2 * n_layers)

layered = build_layered_mesh(layer_zs, bbox, nx=nx, ny=ny,
                             cells_per_layer=cells_per_layer)
print("LayeredMesh: n_cells=%d n_layers=%d mesh.nodes=%d" %
      (layered.n_cells, layered.n_layers, layered.mesh.n_nodes))

# --- 2) material (e.g. 316L-ish) + build-plate clamp (bottom z-face) ----------
mat = LinearElasticIsotropic(E=190e9, nu=0.30)

nz = n_layers * cells_per_layer
ny1 = ny + 1
nz1 = nz + 1
# node_id(i,j,k) = i*(ny+1)*(nz+1) + j*(nz+1) + k ; bottom face k==0
bottom_nodes = []
for i in range(nx + 1):
    for j in range(ny + 1):
        bottom_nodes.append(i * ny1 * nz1 + j * nz1 + 0)
bottom_nodes = np.array(bottom_nodes, dtype=np.int64)
bc_dofs = np.stack([bottom_nodes * 3 + d for d in range(3)], axis=1).ravel()
dirichlet_bcs = [DirichletBC.fixed(jnp.asarray(bc_dofs))]

# --- 3) synthetic per-layer cooling thermal strain (alpha*ΔT, negative) --------
alpha_T = 1.2e-5
dT = -200.0  # 200 K cooling per layer
eps_th = jnp.full((n_layers, layered.n_cells), alpha_T * dT)  # (n_layers, n_cells)

# --- 4) run the true 3D layer-by-layer thermo-mechanical solve ---------------
res = solve_thermomechanical(layered, mat,
                            thermal_strain_per_layer=eps_th,
                            dirichlet_bcs=dirichlet_bcs, dim=3)

U = res.U_final
stress = res.residual_stress
print("U_final shape=%s max|U|=%.3e m" % (U.shape, float(jnp.max(jnp.abs(U)))))
print("residual_stress shape=%s max|σ|=%.3e Pa" %
      (stress.shape, float(jnp.max(jnp.abs(stress)))))
finite = bool(jnp.all(jnp.isfinite(U))) and bool(jnp.all(jnp.isfinite(stress)))
print("ALL FINITE:", finite)
print("NON-TRIVIAL (max|U|>0 and max|σ|>0):",
      bool(jnp.max(jnp.abs(U)) > 0) and bool(jnp.max(jnp.abs(stress)) > 0))
assert finite and jnp.max(jnp.abs(U)) > 0 and jnp.max(jnp.abs(stress)) > 0
print("SMOKE OK: true-3D layered AM thermo-mechanical solve works.")
