import jax, jax.numpy as jnp, time
jax.config.update('jax_enable_x64', True)
from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.materials import get_material
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps
m=get_material('316L'); TL=float(m.T_liquidus); TS=float(m.T_solidus)
BX,BY,BZ=1.2e-3,0.6e-3,0.4e-3; R=100e-6; P=600.0
def coupon(dx): return G.from_sdf_fn(lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[...,0])-BX/2,jnp.abs(x[...,1])-BY/2),jnp.abs(x[...,2])-BZ/2), bounds=[(-BX/2,BX/2),(-BY/2,BY/2),(-BZ/2,BZ/2)], spacing=dx, name='c')
def plan(): return ProcessPlan.uniform(1,modality='SLM',laser_power=P,scan_speed=0.8,layer_thickness=BZ,hatch_spacing=1.4*R,beam_radius=R,absorption=0.45,preheat_temp=400.0)
for model in ('integrated','point'):
  for dx in (100e-6,50e-6,25e-6,12.5e-6):
    g=coupon(dx); p=plan(); ns=suggest_n_steps(g,p)
    t0=time.perf_counter(); th=solve_enthalpy_thermal(geometry=g,process=p,params=dict(material='316L',source_model=model,n_steps=ns)); el=time.perf_counter()-t0
    sol=g.sdf<0; pk=th.peak_temperature; mt=(pk>TL)&sol; c=g.coords()
    Lx=float(jnp.max(c[...,0],where=mt,initial=0.)-jnp.min(c[...,0],where=mt,initial=0.))*1e3
    Ly=float(jnp.max(c[...,1],where=mt,initial=0.)-jnp.min(c[...,1],where=mt,initial=0.))*1e3
    vol=float(jnp.sum(mt))*dx**3*1e9
    print(f'[{model:10s}] dx={dx*1e6:6.1f} nvox={int(g.sdf.size):6d} ns={ns:5d} {el:6.2f}s peak={float(jnp.max(pk)):8.1f} nml={int(jnp.sum(mt)):6d} Lx={Lx:6.3f} Ly={Ly:6.3f} vol={vol:8.4f}mm3',flush=True)
