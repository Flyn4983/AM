"""独立核对 Eagar-Tsai 中心温度：scipy.quad vs 闭式解 vs jax 实现。"""
import sys
sys.path.insert(0, "src")

import numpy as np
from scipy.integrate import quad

from amforge.materials import get_material

mat = get_material("Ti6Al4V")
k = 0.5 * (mat.k_solid + mat.k_liquid)
rho = 0.5 * (mat.rho_solid + mat.rho_liquid)
cp = 0.5 * (mat.cp_solid + mat.cp_liquid)
rho_cp = rho * cp
alpha = k / rho_cp
A, P, rb, v, T0 = 0.4, 220.0, 50e-6, 0.9, 473.0
Pab = A * P
print(f"k={k:.4g}  rho={rho:.4g}  cp={cp:.4g}  alpha={alpha:.4e} m^2/s")

pref = 2.0 * Pab / (rho_cp * np.pi ** 1.5)


def integrand(tau, vv):
    a = 4 * alpha * tau + 0.5 * rb ** 2
    b = 4 * alpha * tau
    return np.exp(-((vv * tau) ** 2) / a) / (a * np.sqrt(b))


res = {}
for vv in (0.0, 0.9):
    # tau = u^2 消除 1/sqrt(tau) 奇点，u 在 [0, inf)
    f = lambda u: 2 * u * integrand(u * u, vv)
    val, err = quad(f, 0, np.inf, limit=400)
    res[vv] = T0 + pref * val
    print(f"v={vv:>4}: scipy T_center = {res[vv]:12.2f} K   (abs err {pref*err:.2e})")

T_exact = T0 + (np.sqrt(2) / 2) * Pab / (rho_cp * np.sqrt(np.pi) * alpha * rb)
print(f"闭式解 (v=0 极限)      = {T_exact:12.2f} K")
print(f"scipy(v=0)/闭式        = {(res[0.0]-T0)/(T_exact-T0):.6f}   <-- 应为 1.000000")
print(f"闭式/scipy(v=0.9)      = {(T_exact-T0)/(res[0.9]-T0):.4f}   <-- 速度效应")

# jax 实现在同一工况下
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from amforge.meltpool import eagar_tsai_field

zero = jnp.zeros((1, 1, 1))
for tm_fac in (40.0, 400.0, 4000.0):
    tau_max = 4.0 * (tm_fac * rb) ** 2 / alpha
    T_num = float(eagar_tsai_field(
        absorbed_power=Pab, scan_speed=v, beam_radius=rb, T0=T0,
        alpha=alpha, rho_cp=rho_cp, X=zero, Y=zero, Z=zero,
        n_quad=200, tau_max=tau_max)[0, 0, 0])
    print(f"jax  tau_max={tau_max:.3e}s  T={T_num:12.2f} K   "
          f"比 scipy(v=0.9) {T_num/res[0.9]:.4f}")
