"""端到端链路冒烟测试：geometry -> ... -> verdict 全链路是否无缝链接。"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import amforge as af
from amforge import geometry as G

print("=== import_status ===")
st = af.import_status()
for k, v in st.items():
    print(f"  {k}: {v}")
print("(空 = 全部就绪)" if not st else "")

print("\n=== 已注册求解器 ===")
print(af.solver_table())

# 小几何：半径 0.45mm 的球，spacing 50µm
part = G.from_sdf_fn(
    lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
    bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="smoke_sphere")

print(f"\n几何: {part.shape} 实体体素 {int(jnp.sum(part.occupancy))}")

pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")
print("\n=== 自动选路 ===")
print(pipe.describe())

out = pipe.run(geometry=part,
               params={"material": "316L", "service_force": 800.0})
verdict = out["verdict"]
print("\n=== 判定结果 ===")
print(verdict.summary())
print("\nmargin_report:", verdict.margin_report)
