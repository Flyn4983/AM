"""Isolate the pressure projection: does _project make a field divergence-free?"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import amforge as af
from amforge import meltpool as MP
from amforge.materials import get_material

mat = get_material("316L")
cfg = MP.MeltPoolConfig(nx=20, ny=14, nz=14)
shape = (20, 14, 14)
dx = cfg.dx
dt = 1e-7


def run(tag, metal):
    rho = 8000.0 * metal + 8.0 * (1.0 - metal)
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, shape + (3,)) * 0.1
    u = u.at[..., 0].add(jnp.linspace(-1.0, 1.0, 20)[:, None, None])
    u = u * metal[..., None]
    d0f = jnp.abs(MP._div_face(u, dx))
    d0 = float(jnp.max(d0f))
    print(f"[{tag}]  div_before(face) = {d0:.4e}")
    for niter in (10, 30, 60, 120, 300):
        u_new, p = MP._project(u, rho, metal, dt, dx, niter)
        dfield = jnp.abs(MP._div_face(u_new * metal[..., None], dx))
        d1 = float(jnp.max(dfield))
        # 线性系统残差：区分"CG 未收敛" vs "算子/修正不相容"
        Aop, Minv, rhs_of, correct = MP._pressure_system(rho, metal, dt, dx)
        b = rhs_of(u)
        res = float(jnp.linalg.norm(Aop(p) - b) / jnp.linalg.norm(b))
        # 残差最大处的位置
        k = int(jnp.argmax(dfield))
        loc = jnp.unravel_index(k, shape)
        print(f"    it={niter:3d} div_after={d1:.3e} ratio={d1/d0:.3e}"
              f" pmax={float(jnp.max(jnp.abs(p))):.3e} lin_res={res:.3e}"
              f" argmax={tuple(int(x) for x in loc)}")
    # 排除域顶/界面层后的内部散度
    interior = jnp.ones(shape).at[:, :, -1].set(0.0).at[:, :, 0].set(0.0)
    interior = interior.at[0].set(0.0).at[-1].set(0.0).at[:, 0].set(0.0).at[:, -1].set(0.0)
    u_new, p = MP._project(u, rho, metal, dt, dx, 300)
    din = jnp.abs(MP._div_face(u_new * metal[..., None], dx)) * interior * metal
    print(f"    interior-only div_after = {float(jnp.max(din)):.3e}"
          f"  ratio={float(jnp.max(din))/d0:.3e}")


run("all-metal", jnp.ones(shape))

zi = jnp.arange(shape[2])
F = (zi < 8).astype(jnp.float64)[None, None, :] * jnp.ones(shape)
run("free-surface", 0.5 * (1.0 + jnp.tanh((F - 0.5) / 0.15)))
print("ISOLATION DONE")
