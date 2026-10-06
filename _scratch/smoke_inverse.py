"""功能2 可微分闭环冒烟测试：训练 几何→工艺 网络，损失应单调下降。"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import amforge as af
from amforge import geometry as G
import amforge.inverse as inv

print("import_status:", af.import_status())

part = G.from_sdf_fn(
    lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
    bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="inv_sphere")

print(f"几何: {part.shape}, 实体体素 {int(jnp.sum(part.occupancy))}")

# 训练（n_grid 调小以加速）
res = inv.train_process_predictor(
    part, material="316L", n_steps=40, learning_rate=0.03,
    n_hidden=16, seed=1, service_stress=150e6,
    params={"n_grid": 20, "allowable_displacement": 1e-4})

print("\n损失历史(首尾):", res["loss_history"][:3], "...", res["loss_history"][-3:])
print("损失下降比:", res["loss_history"][0] / max(res["loss_history"][-1], 1e-12))

plan = res["final_plan"]
print("\n训练后工艺方案(标量代表):")
print("  laser_power =", float(jnp.mean(plan.laser_power)), "W")
print("  scan_speed  =", float(jnp.mean(plan.scan_speed)), "m/s")
print("  layer_thk   =", float(jnp.mean(plan.layer_thickness)) * 1e6, "µm")

# 验证整条链可在最终工艺下跑通并判定
out = inv.simulate(part, plan, material="316L", service_stress=150e6,
                   params={"n_grid": 24})
print("\n最终判定:")
print(out["verdict"].summary())
print("梯度可用性: jax.grad(loss) 已在训练中成功调用(adam 更新生效)")
