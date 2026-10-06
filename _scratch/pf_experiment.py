"""Isolate the anisotropy sign / correction bug in the dendrite phase field."""
import jax
import jax.numpy as jnp
from diffmech.methods.phase_field.phase_field import laplacian, gradient

jax.config.update("jax_enable_x64", True)


def div_2d(fx, fy, dx):
    fxp = jnp.pad(fx, ((1, 1), (1, 1)), mode="edge")
    fyp = jnp.pad(fy, ((1, 1), (1, 1)), mode="edge")
    return ((fxp[1:-1, 2:] - fxp[1:-1, :-2]) / (2 * dx)
            + (fyp[2:, 1:-1] - fyp[:-2, 1:-1]) / (2 * dx))


def step(phi, u, cfg):
    dx = 1.0
    g = gradient(phi, dx, 2, "neumann")
    gx, gy = g[..., 0], g[..., 1]
    theta = jnp.atan2(gy, gx)
    ang = cfg["folds"] * (theta - cfg["theta0"])
    a = 1.0 + cfg["aniso_sign"] * cfg["eps4"] * jnp.cos(ang)
    a2 = a * a
    a2gx = a2 * gx
    a2gy = a2 * gy
    div_a2 = div_2d(a2gx, a2gy, dx)
    da2_dtheta = 2.0 * a * (-cfg["folds"] * cfg["eps4"] * jnp.sin(ang))
    gt = gradient(theta, dx, 2, "neumann")
    dot = gt[..., 0] * gx + gt[..., 1] * gy
    corr = jnp.where(cfg["use_corr"],
                      -0.5 * da2_dtheta * dot * (1.0 - phi ** 2), 0.0)
    theta_k = cfg["eps_k"] * jnp.cos(ang)
    bulk = (phi - phi ** 3 + cfg["lam"] * (1.0 - phi ** 2) ** 2 * (u + theta_k))
    rxn = bulk / (cfg["lam"] * cfg["tau0"])
    dphi = rxn + div_a2 + corr
    return jnp.clip(phi + cfg["dt"] * dphi, -1.0, 1.0)


def run(cfg, n=150):
    nx = ny = 96
    xs = jnp.arange(nx) - (nx - 1) / 2.0
    ys = jnp.arange(ny) - (ny - 1) / 2.0
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    r = jnp.sqrt(X ** 2 + Y ** 2)
    phi0 = jnp.tanh((4.0 - r) / 2.0)
    u = jnp.full((nx, ny), 0.55)
    phi = phi0
    for _ in range(n):
        phi = step(phi, u, cfg)
    # angular solid-mass: axis vs diagonal
    solid = (phi > 0).astype(jnp.float64)
    ang = jnp.mod(jnp.atan2(Y, X), jnp.pi / 2.0)
    mask_arm = (jnp.abs(ang) < (jnp.pi / 8.0)) | (jnp.abs(ang - jnp.pi / 2.0) < (jnp.pi / 8.0))
    mask_diag = jnp.abs(ang - jnp.pi / 4.0) < (jnp.pi / 8.0)
    arm = jnp.sum(solid * mask_arm)
    diag = jnp.sum(solid * mask_diag)
    return arm / jnp.maximum(diag, 1e-12), float(solid.mean())


if __name__ == "__main__":
    base = dict(eps4=0.15, eps_k=0.05, lam=2.0, tau0=0.2, dt=0.1, folds=4, theta0=0.0)
    for aniso_sign in (+1, -1):
        for use_corr in (True, False):
            cfg = dict(base, aniso_sign=aniso_sign, use_corr=use_corr)
            ratio, sf = run(cfg)
            print(f"aniso_sign={aniso_sign:+.0f} use_corr={use_corr} "
                  f"-> arm/diag={ratio:.2f} sf={sf:.3f} "
                  f"{'AXIS (good)' if ratio > 1.3 else ('DIAG (bad)' if ratio < 0.7 else 'ISO')}")
    # eps4=0 isotropic check + ASCII shape
    cfg0 = dict(base, eps4=0.0, aniso_sign=+1, use_corr=False)
    phi_end = None
    nx = ny = 96
    xs = jnp.arange(nx) - (nx - 1) / 2.0
    ys = jnp.arange(ny) - (ny - 1) / 2.0
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    r = jnp.sqrt(X ** 2 + Y ** 2)
    phi = jnp.tanh((4.0 - r) / 2.0)
    u = jnp.full((nx, ny), 0.55)
    for _ in range(150):
        phi = step(phi, u, cfg0)
    ratio0, sf0 = run.__wrapped__(cfg0) if False else (None, None)
    # recompute ratio for eps4=0
    solid = (phi > 0).astype(jnp.float64)
    ang = jnp.mod(jnp.atan2(Y, X), jnp.pi / 2.0)
    mask_arm = (jnp.abs(ang) < (jnp.pi / 8.0)) | (jnp.abs(ang - jnp.pi / 2.0) < (jnp.pi / 8.0))
    mask_diag = jnp.abs(ang - jnp.pi / 4.0) < (jnp.pi / 8.0)
    r0 = jnp.sum(solid * mask_arm) / jnp.maximum(jnp.sum(solid * mask_diag), 1e-12)
    print(f"eps4=0 (isotropic) -> arm/diag={float(r0):.2f} sf={float(solid.mean()):.3f}")
    # ASCII (downsampled)
    stepi = 4
    rows = []
    for i in range(0, ny, stepi):
        row = "".join("#" if phi[i, j] > 0 else ("." if phi[i, j] > -0.5 else " ")
                      for j in range(0, nx, stepi))
        rows.append(row)
    print("\n".join(rows))

