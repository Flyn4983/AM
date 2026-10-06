"""逐损失分量检查梯度 NaN 来源。"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import amforge as af
from amforge import geometry as G
import amforge.inverse as inv
from amforge.materials import get_material

part = G.from_sdf_fn(
    lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
    bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="dbg")
nL = int(part.layer_count(40e-6))
feats = inv.geometry_features(part)
rng = jax.random.PRNGKey(1)
theta = inv.mlp_init(rng, int(feats.shape[0]), 16, n_out=len(inv.PROCESS_BOUNDS))
mat = get_material("316L")
spacing = float(part.spacing)
solid = (part.sdf < 0.0).astype(jnp.float64)
cell_vol = spacing ** part.dim


def forward(θ):
    plan = inv.predict_process(θ, feats, n_layers=nL)
    out = inv.simulate(part, plan, material="316L", service_stress=150e6,
                       params={"n_grid": 20, "allowable_displacement": 1e-4})
    return plan, out


def comp(name, fn):
    g = jax.grad(lambda θ: fn(*forward(θ)))(theta)
    flat = jnp.concatenate([jnp.ravel(x) for x in g.values()])
    print(f"  {name:14s} has_nan={bool(jnp.any(jnp.isnan(flat)))} "
          f"max|g|={float(jnp.max(jnp.abs(flat))):.3e}")


def geom_comp(plan, out):
    return jnp.mean(jnp.abs(out["asbuilt"].sdf - part.sdf) * solid) / spacing


def stress_comp(plan, out):
    rvm = out["asbuilt"].von_mises_residual()
    return jnp.sum(rvm * solid) * cell_vol / \
        jnp.maximum(jnp.sum(solid) * cell_vol, 1e-30) / mat.sigma_y


def strain_comp(plan, out):
    sv = out["asbuilt"].residual_strain
    return jnp.sum(jnp.sqrt(jnp.sum(sv ** 2, axis=-1) + 1e-30) * solid) * cell_vol / \
        jnp.maximum(jnp.sum(solid) * cell_vol, 1e-30)


def defect_comp(plan, out):
    return out["meltpool"].defect_score()


def safety_comp(plan, out):
    return jax.nn.relu(1.0 - out["verdict"].strength_safety_factor)


for nm, f in [("geom", geom_comp), ("stress", stress_comp),
              ("strain", strain_comp), ("defect", defect_comp),
              ("safety", safety_comp)]:
    comp(nm, f)
