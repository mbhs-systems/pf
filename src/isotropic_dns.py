import sys;

sys.path.append('../dedaLES')
import logging
import time

import numpy as np
from numpy import pi

from dedalus.extras import flow_tools
import dedaLES

# import mod_nav_stokes as mns
import channel_mod

startt = time.time()
logger = logging.getLogger(__name__)


def log_magnitude(xmesh, ymesh, data):
    '''
    Log magnitude function for scaling a magnitude for plotting
    complex valueds scalar fields
    '''
    return xmesh, ymesh, np.log10(np.abs(data))


# Parameters
if len(sys.argv) > 1:
    rep = np.power(2, int(sys.argv[1]))
else:
    rep = 8
nx = ny = nz = rep
Lx = Ly = Lz = 2 * pi

# Homoegneous Navier-Stokes equations
closure = None
# model = mns.NavierStokesTriplyPeriodicFlow(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, ν=1.0, closure=closure)
model = channel_mod.ChannelMod_DNS(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, xleft=nx, yleft=ny, zbottom=nz,
                                   nu=1.0, rho=1.0)
print('Number of Eqs:')
print(model.problem.equations)

# Add boundary conditions equivalent to boussinesq example
model.set_bc("nopenetration", "top", "bottom")
model.set_bc("freeslip", "top", "bottom")
# model.problem.add_bc("right(b) = 0")
# model.problem.add_bc("left(b) = Δb")

# print('Number of Eqs:')
# print(len(model.problem.equations))

model.build_solver()

# Random initial condition. Re_k = u k / ν => u ~ ν * Re_k / k
Re = 1000.0  # Re at grid scale
u0 = Re / nx
model.u['g'] = u0 * dedaLES.random_noise(model.domain, seed=23)
model.v['g'] = u0 * dedaLES.random_noise(model.domain, seed=42)

model.u['g'] = model.u['g'] - np.mean(model.u['g'])
model.v['g'] = model.v['g'] - np.mean(model.v['g'])

# Diagnose w from continuity
ux = model.domain.new_field()
vy = model.domain.new_field()
wz = model.domain.new_field()
model.u.differentiate('x', out=ux)
model.v.differentiate('y', out=vy)
model.w.differentiate('z', out=wz)

wz['g'] = - ux['g'] - vy['g']
wz.integrate('z', out=model.w)

# Plot energy spectra...

# Run the simulation
max_u = np.max(model.u['g'])
dt = 0.1 * 2 * pi / (max_u * nx)  # grid-scale turbulence time-scale = 1/(u*k)
model.stop_at(sim_time=np.inf, wall_time=np.inf, iteration=100)

print('Elapsed time: ' + str((time.time() - startt)))

startt = time.time()

# Run the simulation, plot the pressure field occasionally
while model.solver.ok:
    model.solver.step(dt)
print('Elapsed time: ' + str((time.time() - startt)))
