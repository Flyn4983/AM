# DiffMech

A JAX-based **end-to-end differentiable** computational mechanics suite. Every
numerical method is written so that `jax.grad` propagates through the entire
simulation — from initial conditions and material parameters to the final
field state — enabling gradient-based optimisation, inverse design, and
sensitivity analysis without adjoint-code derivation.

All methods support **both 2D and 3D** simulation.

## Numerical methods

| Method | Module | Physics | 2D | 3D |
|--------|--------|---------|-----|-----|
| Finite Element (FEM) | `methods.fem` | Linear & nonlinear solid mechanics (hyperelasticity, plasticity) | ✅ | ✅ |
| Finite Volume (FVM) | `methods.fvm` | Scalar advection, compressible Euler, incompressible Stokes | ✅ | ✅ |
| Discrete Element (DEM) | `methods.dem` | Granular contact dynamics (spring–dashpot, Coulomb friction) | ✅ | ✅ |
| Smoothed Particle Hydrodynamics (SPH) | `methods.sph` | Weakly compressible free-surface flow | ✅ | ✅ |
| Material Point (MPM) | `methods.mpm` | MLS-MPM solids & fluids (PIC/APIC, Neo-Hookean, EOS) | ✅ | ✅ |
| Phase Field | `methods.phase_field` | Allen-Cahn microstructure, variational brittle fracture | ✅ | ✅ |
| Crystal Plasticity FEM (CPFE) | `methods.cpfe` | Rate-dependent crystal plasticity (FCC/BCC slip systems) | ✅ | ✅ |

## Architecture

```
src/diffmech/
├── core/               # Mesh, quadrature, shape functions, fields
│   ├── mesh.py         #   2D (quad4/tri3) & 3D (hex8/tet4) structured meshes
│   ├── quadrature.py   #   Gauss-Legendre, Dunavant, Keaster rules
│   └── shape_functions.py  # tri3, quad4, tet4, hex8
├── materials/          # Differentiable constitutive models
│   ├── linear_elastic.py   # Isotropic Hooke's law
│   ├── hyperelastic.py     # Neo-Hookean
│   └── plasticity.py       # J2 (von Mises) return-mapping
├── solvers/            # Nonlinear & linear solvers
│   ├── newton.py           # Newton-Raphson (while_loop + scan variants)
│   ├── boundary_conditions.py  # Dirichlet BC application
│   └── time_integrators.py    # SSP-RK2, semi-implicit Euler
├── methods/           # Numerical methods (each independently importable)
│   ├── fem/            # Linear & nonlinear FEM
│   ├── fvm/            # Finite volume (advection, Euler, Stokes)
│   ├── dem/            # Discrete element
│   ├── sph/            # Smoothed particle hydrodynamics
│   ├── mpm/            # Material point (MLS-MPM)
│   ├── phase_field/    # Allen-Cahn & phase-field fracture
│   └── cpfe/           # Crystal plasticity FEM
├── preprocess/        # Pre-processing: mesh I/O, boundary detection, load cases
│   ├── mesh_io.py          # VTK/VTU/Gmsh/Abaqus import/export via meshio
│   ├── mesh_tools.py       # Boundary nodes, box-face selection, element volumes
│   ├── load_cases.py       # Uniaxial tension, shear, bending, biaxial, gravity
│   ├── mesh_primitives.py  # Disc, ring, plate-with-hole, L-bracket, cylinder meshes
│   ├── materials.py        # AM alloy library (Ti6Al4V, IN718, AlSi10Mg, 316L)
│   └── problem_builder.py  # Fluent ProblemSetup builder → solve → export
├── postprocess/       # Post-processing: VTK writers, visualisation, probes
│   ├── vtk.py              # VTU/VTK/PVD writers (ParaView), von Mises fields
│   ├── visualization.py    # matplotlib 2D/3D plots (contour, quiver, deformed mesh)
│   ├── probes.py           # Point probes, volume averages, history extraction
│   ├── time_series.py      # PVD animation + NumPy history logger
│   ├── derived_fields.py   # Stress/strain recovery: von Mises, principal, hydrostatic
│   ├── analysis.py         # Error norms, convergence studies, solution comparison
│   └── animation.py        # GIF / PNG field animations (no ffmpeg needed)
└── optimize/           # Inverse-design utilities
    └── inverse.py
```

### Design principles

1. **Pure functions.** Every physics routine is a pure function of JAX arrays.
   State is passed in and returned — no mutable objects, no side effects.

2. **Pytree everywhere.** Configuration and state objects are registered as
   JAX pytrees (`register_pytree_node`), so they thread cleanly through
   `jax.jit`, `jax.vmap`, and `jax.grad`.

3. **Scan over loops.** Time-stepping and Newton iterations use
   `jax.lax.scan` (fixed iteration count) for reverse-mode-AD compatibility,
   or `jax.lax.while_loop` (early termination) when differentiability is not
   required.

4. **Autodiff for tangents.** Algorithmic tangents (e.g. `dσ/dε`) are
   computed with `jax.jacrev` rather than hand-derived, guaranteeing
   consistency with the stress implementation.

5. **Unified 2D/3D.** Each method infers dimensionality from array shapes
   and uses dimension-agnostic stencils (`jnp.roll`, `jnp.pad`, `einsum`),
   so the same code path works in 2D and 3D.

## Differentiability

The end-to-end gradient flows through:

- **Linear solves** — `jnp.linalg.solve` is differentiable.
- **Time integration** — `jax.lax.scan` supports reverse-mode AD.
- **Constitutive updates** — explicit Euler sub-stepping (CPFE, MPM, SPH)
  or semi-implicit schemes (phase field, FVM) are AD-friendly.
- **Newton solvers** — the scan-based `newton_solve_diff` variant supports
  `jax.grad` through the nonlinear solve.

Example — optimising the initial CRSS `g0` of a crystal-plasticity
simulation against a final plastic-strain objective:

```python
import jax
import jax.numpy as jnp
from diffmech.core import rectangular_mesh2d
from diffmech.methods.cpfe import (
    CrystalPlasticity, CPFEProblem, solve_cpfe,
    average_equivalent_plastic_strain,
)

def objective(g0):
    b = jnp.array([[1.0, 0.0]])
    n = jnp.array([[0.0, 1.0]])
    mat = CrystalPlasticity(
        E=1000.0, nu=0.3, slip_directions=b, slip_normals=n,
        gamma_dot0=0.01, m=0.5, g0=g0, h0=10.0, g_sat=100.0,
    )
    mesh = rectangular_mesh2d(nx=2, ny=2)
    # ... set up Dirichlet BCs ...
    problem = CPFEProblem(mesh=mesh, material=mat, dirichlet_bcs=bcs, dim=2)
    sol = solve_cpfe(problem, jnp.linspace(0.0, 1.0, 20), dt=0.1, n_sub=5)
    return average_equivalent_plastic_strain(sol.state)

grad = jax.grad(objective)(jnp.array(1.0))  # d(objective)/d(g0)
```

## Pre- & post-processing

DiffMech ships a convenient pre/post-processing toolkit so a full simulation
workflow — mesh in, results out — needs no external glue code.

### Pre-processing (`diffmech.preprocess`)

- **Mesh I/O** — read/write VTK, VTU, Gmsh `.msh`, Abaqus `.inp`, Exodus,
  XDMF, … via `meshio`. Structured-mesh constructors for 2D (quad4/tri3) and
  3D (hex8/tet4).
- **Mesh tools** — boundary-node detection, box-face / box-interior node
  selection, element volumes, multi-material region assignment, structured
  refinement.
- **Load cases** — one-liner generators for uniaxial tension/compression,
  simple shear, biaxial, pure bending, gravity, and clamped-face BCs.

```python
from diffmech.preprocess import (
    structured_quad_mesh_2d, uniaxial_tension, write_mesh,
)

mesh = structured_quad_mesh_2d(16, 16, lx=2.0, ly=1.0)
case = uniaxial_tension(mesh, axis=0, strain=0.01)
write_mesh("part.vtu", mesh)   # export for ParaView / Gmsh / Abaqus
```

#### Fluent problem builder

For everyday linear-elastic solves the `ProblemSetup` builder collapses the
mesh → material → load-case → solve → export pipeline into a single fluent
expression, with automatic stress/strain recovery and a no-solve mesh
preview for catching BC mistakes early:

```python
from diffmech.preprocess import ProblemSetup, plate_with_hole_2d, get_material

result = (ProblemSetup()
          .mesh(plate_with_hole_2d(nx=32, ny=32))
          .material(get_material("Ti6Al4V"))      # AM alloy library
          .tension(axis=0, strain=0.005)
          .solve()                                # → U, strain, stress
          .export("plate.vtu"))                    # von Mises, principals, …
result.summary()   # prints node/cell/|U|/von Mises stats
```

`.preview("bc.png")` draws the mesh with clamped (red) vs free (blue) nodes
without running a solve.

### Post-processing (`diffmech.postprocess`)

- **VTK/VTU/PVD writers** — ParaView-ready output with point & cell data,
  deformed-mesh export, and a `PVDSeries` time-animation writer.
- **Visualisation** — quick matplotlib 2D field/contour/quiver/deformed-mesh
  plots and 3D scatter/slice plots.
- **Probes** — point interpolation, volume averages & integrals, boundary
  integrals, field statistics, and time-history extraction.
- **Time series** — combined `TimeSeriesWriter` that emits a ParaView
  animation *and* a NumPy `.npz` history log in one call.
- **Derived fields** — differentiable stress/strain recovery from a
  displacement solution: von Mises stress, principal stresses & directions,
  hydrostatic pressure, equivalent strain (`recover_strain_stress` in
  `methods.fem` + wrappers in `postprocess.derived_fields`).
- **Animation** — `save_field_gif` turns a stack of 2-D fields into a
  self-contained `.gif` (Pillow writer, no ffmpeg) for quick sharing;
  `save_field_grid_png` makes a snapshot grid for reports.

```python
from diffmech.postprocess import FieldCollection, write_vtu, TimeSeriesWriter

fields = FieldCollection(
    point_data={"displacement": U.reshape(n_nodes, dim)},
    cell_data={"von_mises": vm_stress},
)
write_vtu("solution.vtu", mesh, fields, deformed=True)

with TimeSeriesWriter("run", directory="out") as w:
    for t in times:
        w.write(t, mesh, fields, energy=E)
# out/run.pvd + out/run.npz now exist
```

Stress recovery from a FEM displacement field (differentiable, so it can
sit inside a `jax.grad` loss):

```python
from diffmech.methods.fem import recover_strain_stress
from diffmech.postprocess import von_mises_from_stress, stress_field_collection

strain, stress = recover_strain_stress(mesh, U, mat)   # per-cell tensors
vm = von_mises_from_stress(stress)                      # (n_cells,) scalar
write_vtu("result.vtu", mesh,
          fields=stress_field_collection(mesh, U, mat))  # vm + principals + …
```

## End-to-end AM pipeline example

`examples/am_pipeline.py` demonstrates the full differentiable
"process → microstructure → constitutive → structural response → service
performance" chain targeted at **AM digital verification**:

| Stage | Method | Maps |
|-------|--------|------|
| 0. Process → thermal history | FVM heat diffusion + Gaussian laser | laser power → temperature → cooling rate |
| 1. Process → microstructure | Allen-Cahn phase field | cooling rate → grain structure |
| 2. Microstructure → constitutive | Hall-Petch + CPFE | grain size → initial CRSS → plastic strain |
| 3. Constitutive → structural response | Linear-elastic FEM | plastic strain → degraded stiffness → displacement |
| 4. Structural response → service performance | Phase-field fracture | displacement → fracture energy |
| 5. Inverse design | `jax.grad` + gradient descent | optimise laser power to minimise fracture energy |

The entire chain is differentiable — `jax.grad(loss_fn)` propagates from the
final fracture-energy metric all the way back to the AM laser power, enabling
gradient-based inverse design of process parameters.

```bash
python examples/am_pipeline.py
# → writes thermal_history.vtu, microstructure.vtu, structural_response.vtu,
#   fracture.vtu, am_pipeline.npz (optimisation history) to am_pipeline_out/
```

## Installation

```bash
# from source (editable)
pip install -e ".[dev]"
```

Dependencies: JAX ≥ 0.4.30, NumPy ≥ 1.26, SciPy, matplotlib, optax, meshio.

## Running tests

```bash
pytest tests/ -q
```

Test suite (235 tests) covers:

- Patch tests, body-force cantilever, differentiability (FEM)
- Advection, Euler shock tube, Stokes flow, CFL (FVM)
- Contact forces, energy conservation, 2D/3D (DEM)
- Pressure, momentum conservation, differentiability (SPH)
- Free-fall, volume conservation, PIC/APIC, 2D/3D (MPM)
- Allen-Cahn spinodal decomposition, phase-field fracture growth (Phase Field)
- Slip systems, constitutive update, 2D shear / 3D tension, gradients (CPFE)
- Mesh I/O round-trips, boundary detection, load cases, element volumes (preprocess)
- VTU/VTK/PVD writers, probes, time-series, visualisation smoke tests (postprocess)
- End-to-end AM pipeline forward pass, differentiability, inverse design (am_pipeline)

## Context

DiffMech targets **additive-manufacturing (AM) digital verification** —
mapping the full "process → microstructure → constitutive → structural
response → service performance" chain with differentiable physics:

- **Phase field** models microstructure evolution (grain growth, solidification).
- **Crystal plasticity FEM** links microstructure to macroscopic yield.
- **FEM / MPM** predict structural response under service loads.
- **Phase-field fracture** predicts failure and service life.
- End-to-end gradients enable inverse design of process parameters.

## License

MIT
