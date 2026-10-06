"""核对正切代换版 eagar_tsai_field：闭式解 + scipy 参照 + 收敛性 + tau_max 免疫。"""
import sys
sys.path.insert(0, "src")

import numpy as np
from scipy.integrate import quad

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from amforge.materials import get_material
from amforge.meltpool import eagar_tsai_field


def props(name):
    m = get_material(name)
    k = 0.5 * (m.k_solid + m.k_liquid)
    rho = 0.5 * (m.rho_solid + m.rho_liquid)
    cp = 0.5 * (m.cp_solid + m.cp_liquid)
    return k, rho * cp, k / (rho * cp)


def scipy_ref(Pab, v, rb, T0, alpha, rho_cp, x, y, z):
    """原始 τ 积分，用 tau=u^2 消奇点后交给 QUADPACK 自适应求积。"""
    pref = 2.0 * Pab / (rho_cp * np.pi ** 1.5)

    def f(u):
        tau = u * u
        a = 4 * alpha * tau + 0.5 * rb ** 2
        b = 4 * alpha * tau
        if b == 0.0:
            return 0.0
        e = -((x + v * tau) ** 2 + y ** 2) / a - z ** 2 / b
        return 2 * u * np.exp(e) / (a * np.sqrt(b))

    val, err = quad(f, 0, np.inf, limit=500, epsabs=1e-14, epsrel=1e-12)
    return T0 + pref * val, pref * err


def jax_at(Pab, v, rb, T0, alpha, rho_cp, x, y, z, n_quad=32):
    g = lambda t: jnp.asarray([[[float(t)]]])
    return float(eagar_tsai_field(
        absorbed_power=Pab, scan_speed=v, beam_radius=rb, T0=T0,
        alpha=alpha, rho_cp=rho_cp, X=g(x), Y=g(y), Z=g(z),
        n_quad=n_quad)[0, 0, 0])


print("=" * 78)
print("1) 静止极限 v=0，中心点 —— 应与闭式解 (√2/2)AP/(ρc√πα r_b) 精确相符")
print("=" * 78)
for name in ("Ti6Al4V", "316L", "IN718", "AlSi10Mg"):
    k, rho_cp, alpha = props(name)
    Pab, rb, T0 = 88.0, 50e-6, 353.0
    exact = T0 + (np.sqrt(2) / 2) * Pab / (rho_cp * np.sqrt(np.pi) * alpha * rb)
    for nq in (4, 32):
        num = jax_at(Pab, 0.0, rb, T0, alpha, rho_cp, 0, 0, 0, n_quad=nq)
        print(f"  {name:9s} n_quad={nq:3d}: {num:12.4f} K  闭式 {exact:12.4f} K  "
              f"相对偏差 {abs(num/exact-1):.3e}")

print()
print("=" * 78)
print("2) 移动源多点比对 scipy 自适应求积（含尾迹/侧向/深度方向）")
print("=" * 78)
k, rho_cp, alpha = props("Ti6Al4V")
Pab, rb, T0 = 0.4 * 220.0, 50e-6, 473.0
worst = 0.0
for v in (0.2, 0.9, 2.5):
    for (x, y, z) in [(0, 0, 0), (-1e-4, 0, 0), (-3e-4, 0, 0),
                      (0, 6e-5, 0), (0, 0, -4e-5), (-1.5e-4, 5e-5, -3e-5),
                      (8e-5, 0, 0)]:
        ref, err = scipy_ref(Pab, v, rb, T0, alpha, rho_cp, x, y, z)
        num = jax_at(Pab, v, rb, T0, alpha, rho_cp, x, y, z, n_quad=32)
        rel = abs(num / ref - 1)
        worst = max(worst, rel)
        print(f"  v={v:4.1f}  (x,y,z)=({x*1e6:7.1f},{y*1e6:5.1f},{z*1e6:6.1f})µm  "
              f"jax {num:10.2f}  scipy {ref:10.2f}  rel {rel:.2e}")
print(f"  --> 最大相对偏差 {worst:.3e}")

print()
print("=" * 78)
print("3) 收敛性：n_quad 8→128（对照 256 点）")
print("=" * 78)
L = 1.5e-4
xs = jnp.linspace(-6 * L, 1.5 * L, 10)
ys = jnp.linspace(-2.5 * L, 2.5 * L, 10)
zs = jnp.linspace(0.0, -2.5 * L, 10)
X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
k5, rc5, al5 = props("316L")
fld = lambda n: eagar_tsai_field(absorbed_power=68.0, scan_speed=0.8,
                                 beam_radius=50e-6, T0=353.0, alpha=al5,
                                 rho_cp=rc5, X=X, Y=Y, Z=Z, n_quad=n)
ref = fld(256)
for n in (8, 16, 24, 32, 48, 64, 128):
    rel = float(jnp.max(jnp.abs(fld(n) - ref) / jnp.abs(ref)))
    print(f"  n_quad={n:4d}: 全场最大相对偏差 {rel:.3e}")

print()
print("=" * 78)
print("4) tau_max 免疫性：旧接口传任意值都不应改变结果")
print("=" * 78)
base = jax_at(Pab, 0.9, rb, T0, alpha, rho_cp, 0, 0, 0)
for tm in (None, 1e-6, 1.0, 1e6):
    g = lambda t: jnp.asarray([[[float(t)]]])
    val = float(eagar_tsai_field(
        absorbed_power=Pab, scan_speed=0.9, beam_radius=rb, T0=T0,
        alpha=alpha, rho_cp=rho_cp, X=g(0), Y=g(0), Z=g(0),
        n_quad=32, tau_max=tm)[0, 0, 0])
    print(f"  tau_max={str(tm):>8s}: {val:12.4f} K   diff {abs(val-base):.2e}")
