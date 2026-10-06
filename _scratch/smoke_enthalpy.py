"""smoke：thermal.enthalpy 相变物理验证（P2-①）。"""
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from amforge.geometry import from_sdf_fn
from amforge.core.contracts import ProcessPlan
from amforge.materials import get_material
from amforge.thermal_enthalpy import (
    liquid_fraction, temperature_of_enthalpy, solve_enthalpy_thermal,
    _build_scan_positions, _moving_source, _div_alpha_grad, effective_diffusivity,
)
from dataclasses import replace

# 小箱体几何（网格可比光斑解析：dx < r）
part = from_sdf_fn(
    lambda x: jnp.linalg.norm(x, axis=-1) - 0.30e-3,
    bounds=[(-0.3e-3, 0.3e-3)] * 3, spacing=50e-6, name="box",
)
plan = ProcessPlan.uniform(3, modality="SLM", laser_power=1200.0,
                           scan_speed=0.8, layer_thickness=50e-6,
                           hatch_spacing=120e-6, beam_radius=100e-6,
                           absorption=0.45, preheat_temp=400.0)

mat = get_material("316L")
print("MAT 316L: Ts/Tl/L =", mat.T_solidus, mat.T_liquidus, mat.latent_fusion)
print("GRID shape =", part.sdf.shape, "dx =", part.spacing)

# (a) 液相关联 + 逆映射单调性
T = jnp.linspace(300.0, 2000.0, 200)
f = liquid_fraction(T, mat.T_solidus, mat.T_liquidus)
H = mat.rho_solid * mat.cp_solid * (T - mat.T_ambient) + mat.rho_solid * mat.latent_fusion * f
Tback = temperature_of_enthalpy(H, rho=mat.rho_solid, cp=mat.cp_solid,
                                L=mat.latent_fusion, T_amb=mat.T_ambient,
                                T_sol=mat.T_solidus, T_liq=mat.T_liquidus, n_iter=16)
print("T(H) 逆映射最大误差 =", float(jnp.max(jnp.abs(T - Tback))), "K  (应≈0)")
# 糊状区平台：H 在潜热区间时 T 不应越出 [Ts,Tl]
mushy = (f > 0.01) & (f < 0.99)
print("糊状区 H 区间内 T 范围 = [%.1f, %.1f] K  (应夹在 %.0f~%.0f)"
      % (float(jnp.min(T[mushy])), float(jnp.max(T[mushy])),
         mat.T_solidus, mat.T_liquidus))

# (b) 含潜热 vs 不含潜热：同热源下峰值温度应更低（潜热吸收能量 → 温度更低）
coords = part.coords(); mask = (part.sdf < 0.0).astype(jnp.float64)
rho, cp, k = float(mat.rho_solid), float(mat.cp_solid), float(mat.k_solid)
L = float(mat.latent_fusion)
Ts, Tl, Tamb = float(mat.T_solidus), float(mat.T_liquidus), float(mat.T_ambient)
dx = float(part.spacing)
P = float(jnp.mean(plan.laser_power)); r = float(jnp.mean(plan.beam_radius))
eta = float(jnp.mean(plan.absorption))
dp = max(r*1.2, dx); Q0 = eta*P/(3.14159*r*r*(2*3.14159)**0.5*dp)
dt = 0.35*dx*dx/(2*3* (k/(rho*cp)))
ns = int(_build_scan_positions(coords, plan, dx).shape[0])
pos = _build_scan_positions(coords, plan, dx)
H0 = jnp.zeros_like(mask)

def rhs(H, t, useL):
    T = temperature_of_enthalpy(H, rho=rho, cp=cp, L=(L if useL else 0.0),
                                T_amb=Tamb, T_sol=Ts, T_liq=Tl)
    a = effective_diffusivity(T, k=k, rho=rho, cp=cp, L=(L if useL else 0.0),
                              T_sol=Ts, T_liq=Tl)
    Q = _moving_source(pos[t], coords, Q0, r, dp) * mask
    return _div_alpha_grad(H, a, dx) + Q - 0.5*(T-Tamb)*mask

def run(useL):
    def body(c, t):
        H, peak = c
        T = temperature_of_enthalpy(H, rho=rho, cp=cp, L=(L if useL else 0.0),
                                    T_amb=Tamb, T_sol=Ts, T_liq=Tl)
        k1 = rhs(H, t, useL); H1 = H + dt*k1; k2 = rhs(H1, t, useL)
        Hn = H + 0.5*dt*(k1+k2)
        Tn = temperature_of_enthalpy(Hn, rho=rho, cp=cp, L=(L if useL else 0.0),
                                     T_amb=Tamb, T_sol=Ts, T_liq=Tl)
        return (Hn, jnp.maximum(peak, Tn)), None
    (HL, peakL), _ = jax.lax.scan(body, (H0, jnp.zeros_like(mask)), jnp.arange(ns))
    return HL, peakL

HL, peakL = run(True)
HL0, peakL0 = run(False)
peak_with_L = float(jnp.max(peakL))
peak_no_L = float(jnp.max(peakL0))
print("peak (有潜热) =", peak_with_L, "K   peak (无潜热) =", peak_no_L, "K")
print("潜热吸收使峰值更低?", peak_with_L < peak_no_L)
print("FINITE?", bool(jnp.all(jnp.isfinite(HL)) and jnp.all(jnp.isfinite(HL0))))
# 同时验证封装求解器
th = solve_enthalpy_thermal(geometry=part, process=plan,
                            params={"material": "316L", "n_steps": 48})
print("封装求解器 peak =", float(jnp.max(th.peak_temperature)),
      "fields finite?", bool(jnp.all(jnp.isfinite(th.peak_temperature)) and
      jnp.all(jnp.isfinite(th.final_temperature))))
