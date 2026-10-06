"""Material library for AM alloys (differentiable parameter sets).

This module ships literature-typical mechanical and thermal parameters for
the most common AM alloys — Ti6Al4V, Inconel 718, AlSi10Mg, 316L stainless
steel, and a few others — packaged as :class:`MaterialParams` dataclasses.

All numerical parameters are stored as plain floats so that they can be
freely converted to JAX tracer leaves via :func:`as_jax` (or directly passed
to ``jax.grad`` objectives). The dataclass is *not* frozen, so users can
override individual parameters (e.g. fit them via inverse design) without
rebuilding the whole object.

Sources
-------
The values are representative literature ranges for as-built / HIP AM parts
and are intended for benchmarks and examples, **not** for certification. Real
AM properties are sensitive to machine, parameters, and post-processing;
always validate against experimental data.

Mechanical: E, nu, sigma_y, hardening parameters (Voce).
Thermal:    k, rho, cp -> alpha = k/(rho*cp); rho_cp = rho*cp.
CPFE:       representative initial CRSS `g0`, saturation `g_sat`, hardening `h0`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import jax.numpy as jnp


@dataclass
class MaterialParams:
    """A differentiable material-parameter set for one AM alloy.

    Mechanical (small-strain + J2 plasticity)
    -----------------------------------------
    E : float           Young's modulus [Pa]
    nu : float          Poisson ratio [-]
    sigma_y : float     Initial yield stress [Pa]
    K_hard : float      Linear hardening modulus [Pa]
    sigma_sat : float   Saturation stress (Voce) [Pa]
    eps0 : float        Reference plastic strain for Voce hardening [-]

    Thermal (heat-diffusion / AM thermal stage)
    --------------------------------------------
    k_cond : float      Thermal conductivity [W/(m·K)]
    rho : float         Density [kg/m³]
    cp : float          Specific heat capacity [J/(kg·K)]
    T_melt : float      Melting temperature [K]
    T_amb : float       Ambient / build-plate temperature [K]

    Crystal-plasticity (representative FCC/BCC values)
    --------------------------------------------------
    g0 : float          Initial CRSS [Pa]
    g_sat : float       Saturation CRSS [Pa]
    h0 : float          Hardening modulus [Pa]
    gamma_dot0 : float  Reference shear-strain rate [1/s]
    m_rate : float      Rate-sensitivity exponent [-]

    Metadata
    --------
    name : str          Alloy identifier.
    lattice : str       Crystal structure ("fcc"/"bcc"/"hcp").
    notes : str         Free-form provenance / caveats.
    """

    name: str = ""
    E: float = 0.0
    nu: float = 0.0
    sigma_y: float = 0.0
    K_hard: float = 0.0
    sigma_sat: float = 0.0
    eps0: float = 0.0

    k_cond: float = 0.0
    rho: float = 0.0
    cp: float = 0.0
    T_melt: float = 0.0
    T_amb: float = 300.0

    g0: float = 0.0
    g_sat: float = 0.0
    h0: float = 0.0
    gamma_dot0: float = 0.001
    m_rate: float = 0.05

    lattice: str = ""
    notes: str = ""

    # --- derived -------------------------------------------------------
    @property
    def lam(self) -> float:
        """First Lamé constant."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        """Shear modulus (second Lamé constant)."""
        return self.E / (2 * (1 + self.nu))

    @property
    def alpha(self) -> float:
        """Thermal diffusivity ``k/(rho cp)`` [m²/s]."""
        return self.k_cond / (self.rho * self.cp)

    @property
    def rho_cp(self) -> float:
        """Volumetric heat capacity ``rho * cp`` [J/(m³·K)]."""
        return self.rho * self.cp

    def as_jax(self, dtype=None):
        """Return a :class:`MaterialParams` whose numeric leaves are JAX arrays.

        This is convenient for passing the material into a JIT/grad function
        that should differentiate w.r.t. its parameters.
        """
        kw = asdict(self)
        return _JaxMaterialParams(**{k: jnp.asarray(v, dtype=dtype) if isinstance(v, (int, float)) else v
                                      for k, v in kw.items()})


@dataclass
class _JaxMaterialParams:
    """JAX-array variant of :class:`MaterialParams` (for autodiff)."""
    name: object = None
    E: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    nu: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    sigma_y: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    K_hard: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    sigma_sat: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    eps0: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    k_cond: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    rho: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    cp: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    T_melt: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    T_amb: jnp.ndarray = field(default_factory=lambda: jnp.asarray(300.0))
    g0: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    g_sat: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    h0: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.0))
    gamma_dot0: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.001))
    m_rate: jnp.ndarray = field(default_factory=lambda: jnp.asarray(0.05))
    lattice: object = None
    notes: object = None

    # Derived properties (mirror MaterialParams; work on JAX arrays).
    @property
    def lam(self) -> jnp.ndarray:
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> jnp.ndarray:
        return self.E / (2 * (1 + self.nu))

    @property
    def alpha(self) -> jnp.ndarray:
        return self.k_cond / (self.rho * self.cp)

    @property
    def rho_cp(self) -> jnp.ndarray:
        return self.rho * self.cp


# ---------------------------------------------------------------------------
# Library of common AM alloys
# ---------------------------------------------------------------------------
AM_MATERIALS: dict[str, MaterialParams] = {
    "Ti6Al4V": MaterialParams(
        name="Ti6Al4V",
        E=110e9, nu=0.31, sigma_y=950e6, K_hard=200e6, sigma_sat=1200e6, eps0=0.01,
        k_cond=6.7, rho=4430.0, cp=526.0, T_melt=1878.0, T_amb=300.0,
        g0=180e6, g_sat=450e6, h0=120e6, gamma_dot0=0.001, m_rate=0.05,
        lattice="hcp",
        notes="Ti-6Al-4V, as-built LPBF representative values. Strong "
              "texture anisotropy; properties vary with build orientation.",
    ),
    "IN718": MaterialParams(
        name="IN718",
        E=200e9, nu=0.29, sigma_y=1100e6, K_hard=1500e6, sigma_sat=1450e6, eps0=0.015,
        k_cond=11.4, rho=8190.0, cp=435.0, T_melt=1610.0, T_amb=300.0,
        g0=320e6, g_sat=650e6, h0=250e6, gamma_dot0=0.001, m_rate=0.05,
        lattice="fcc",
        notes="Inconel 718, nickel superalloy. Solution + aging gives the "
              "high strength; as-built is lower. High-temperature service.",
    ),
    "AlSi10Mg": MaterialParams(
        name="AlSi10Mg",
        E=75e9, nu=0.33, sigma_y=270e6, K_hard=400e6, sigma_sat=350e6, eps0=0.01,
        k_cond=150.0, rho=2670.0, cp=920.0, T_melt=870.0, T_amb=300.0,
        g0=80e6, g_sat=180e6, h0=60e6, gamma_dot0=0.001, m_rate=0.05,
        lattice="fcc",
        notes="AlSi10Mg casting/AM alloy. High thermal conductivity makes it "
              "a common AM heat-sink / structural material.",
    ),
    "316L": MaterialParams(
        name="316L",
        E=193e9, nu=0.30, sigma_y=460e6, K_hard=1200e6, sigma_sat=700e6, eps0=0.02,
        k_cond=16.0, rho=7990.0, cp=500.0, T_melt=1670.0, T_amb=300.0,
        g0=120e6, g_sat=300e6, h0=200e6, gamma_dot0=0.001, m_rate=0.05,
        lattice="fcc",
        notes="316L austenitic stainless steel. Good ductility, work-hardening, "
              "corrosion resistance; common AM medical/marine alloy.",
    ),
    "IN625": MaterialParams(
        name="IN625",
        E=208e9, nu=0.31, sigma_y=725e6, K_hard=900e6, sigma_sat=1000e6, eps0=0.015,
        k_cond=9.8, rho=8440.0, cp=410.0, T_melt=1623.0, T_amb=300.0,
        g0=250e6, g_sat=520e6, h0=180e6, gamma_dot0=0.001, m_rate=0.05,
        lattice="fcc",
        notes="Inconel 625 nickel superalloy. Excellent fatigue and "
              "creep strength; common in aerospace / energy AM parts.",
    ),
    "SS316": MaterialParams(  # alias kept for convenience
        name="316L-stress-relieved",
        E=193e9, nu=0.30, sigma_y=520e6, K_hard=1000e6, sigma_sat=680e6, eps0=0.02,
        k_cond=16.0, rho=7990.0, cp=500.0, T_melt=1670.0, T_amb=300.0,
        g0=120e6, g_sat=300e6, h0=200e6, gamma_dot0=0.001, m_rate=0.05,
        lattice="fcc",
        notes="Stress-relieved 316L (higher yield than as-built).",
    ),
}


class MaterialLibrary:
    """Convenience accessor over the bundled AM material database.

    Example
    -------
    >>> lib = MaterialLibrary()
    >>> ti = lib["Ti6Al4V"]
    >>> ti.E, ti.sigma_y, ti.alpha
    >>> ti_jax = ti.as_jax()           # for autodiff
    """

    def __init__(self, materials: dict[str, MaterialParams] | None = None):
        self._materials = dict(materials) if materials else dict(AM_MATERIALS)

    def __getitem__(self, name: str) -> MaterialParams:
        if name not in self._materials:
            raise KeyError(
                f"material {name!r} not in library; available: "
                f"{sorted(self._materials)}"
            )
        return self._materials[name]

    def __contains__(self, name: str) -> bool:
        return name in self._materials

    def names(self) -> list[str]:
        return sorted(self._materials)

    def add(self, material: MaterialParams) -> None:
        """Register a new (or override an existing) material."""
        if not material.name:
            raise ValueError("MaterialParams.name must be set")
        self._materials[material.name] = material

    def __iter__(self):
        return iter(self._materials.values())

    def __len__(self) -> int:
        return len(self._materials)


def get_material(name: str) -> MaterialParams:
    """Module-level shortcut: ``get_material("Ti6Al4V")``."""
    return MaterialLibrary()[name]


__all__ = [
    "MaterialParams", "MaterialLibrary", "AM_MATERIALS", "get_material",
]
