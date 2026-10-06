"""调试：检查 inverse 训练中的 NaN 来源（梯度 vs 前向）。"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import amforge as af
from amforge import geometry as G
import amforge.inverse as inv

part = G.from_sdf_fn(
    lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
    bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="dbg")

feats = inv.geometry_features(part)
rng = jax.random.PRNGKey(1)
theta = inv.mlp_init(rng, int(feats.shape[0]), 16, n_out=len(inv.PROCESS_BOUNDS))

# 前向：检查各契约是否含 NaN
plan = inv.predict_process(theta, feats, n_layers=int(part.layer_count(40e-6)))
out = inv.simulate(part, plan, material="316L", service_stress=150e6,
                   params={"n_grid": 20, "allowable_displacement": 1e-4})
for k, v in out.items():
    if hasattr(v, "temperature"):
        arr = v.temperature
    elif hasattr(v, "sdf"):
        arr = v.sdf
    elif hasattr(v, "von_mises"):
        arr = v.von_mises
    elif hasattr(v, "peak_temperature"):
        arr = v.peak_temperature
    elif hasattr(v, "grain_size"):
        arr = v.grain_size
    elif hasattr(v, "E_inplane"):
        arr = v.E_inplane
    else:
        arr = None
    if arr is not None:
        print(f"  {k:12s} has_nan={bool(jnp.any(jnp.isnan(arr)))} "
              f"has_inf={bool(jnp.any(jnp.isinf(arr)))} "
              f"max={float(jnp.max(jnp.abs(arr))):.3e}")

# 梯度统计
loss_and_grad = jax.value_and_grad(inv.loss_fn)
loss, grads = loss_and_grad(theta, part, material="316L", service_stress=150e6,
                            params={"n_grid": 20, "allowable_displacement": 1e-4})
print(f"\nloss = {float(loss):.4e}")
flat = jnp.concatenate([jnp.ravel(g) for g in grads.values()])
print(f"grad: has_nan={bool(jnp.any(jnp.isnan(flat)))} "
      f"has_inf={bool(jnp.any(jnp.isinf(flat)))} "
      f"max={float(jnp.max(jnp.abs(flat))):.3e} "
      f"mean={float(jnp.mean(jnp.abs(flat))):.3e}")

# 逐参数块
for k, g in grads.items():
    print(f"  {k:4s} max|g|={float(jnp.max(jnp.abs(g))):.3e}")
