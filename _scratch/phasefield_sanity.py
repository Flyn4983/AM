"""Phase-field: find clean young-dendrite config + rotation(G-orientation) test."""
import jax
import jax.numpy as jnp

from amforge.phasefield import (
    DendriteConfig, init_isothermal, run_phase_field,
    solid_fraction, arm_symmetry_4fold,
)
jax.config.update("jax_enable_x64", True)


def run(theta0, n, grid=(96, 96), undercool=0.55, eps4=0.15):
    cfg = DendriteConfig(eps4=eps4, principal_axis=theta0, coupled_thermal=False)
    init = init_isothermal(grid, undercool, seed_radius=4.0)
    phi = run_phase_field(cfg, init, n_steps=n)["phi"]
    return float(solid_fraction(phi)), float(arm_symmetry_4fold(phi))


if __name__ == "__main__":
    print("sweep n (theta0=0): sf and axis/diag tip ratio")
    for n in (50, 70, 90, 110, 130):
        sf, ratio = run(0.0, n)
        print(f"  n={n:3d} sf={sf:.3f} ratio={ratio:.2f} "
              f"{'4FOLD' if ratio > 1.2 else ('diag' if ratio < 0.8 else 'iso')}")
    print("\nrotation test at n=90 (G-orientation: arms follow theta0):")
    for th in (0.0, jnp.pi/4, jnp.pi/2):
        sf, ratio = run(th, 90)
        lab = {0.0: "axis", jnp.pi/4: "diag45", jnp.pi/2: "axis90"}[th]
        print(f"  theta0={lab} sf={sf:.3f} ratio={ratio:.2f}")
    print("  expect: ratio(0)~ratio(90)>1 (arms on axes); ratio(45)<1 (arms on diagonals)")
