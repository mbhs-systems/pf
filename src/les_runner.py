import sys;

sys.path.append('../dedaLES')
import logging
import time

import numpy as np
from numpy import pi

from dedalus.extras import flow_tools
from dedalus.extras import plot_tools

import dedaLES

import src.dns_model as dns


def fourier_filter(k, length_scale):
    '''
    Evaluate a Gaussian LES filter for a "wavenumber" input vector k

    :param k: input vector of spatial wave numbers
    :param length_scale: length scale (\Delta)
    :return: filter output
    '''
    return np.exp(-np.power(2, k) * np.power(2, length_scale) / 24)


def filter_field(field, length_scale):
    '''Bandpass FFT filter of the form <0, 0, ....., 0, 1, 1, ...., 1, 0, 0, ....>'''

    # spec = np.fft.fftn(field['g'])
    spec = field['c']
    xfb = np.linspace(0, 1, field['c'].shape[0])
    yfb = np.linspace(0, 1, field['c'].shape[1])
    zfb = np.linspace(0, 1, field['c'].shape[2])
    fgrid = np.meshgrid(xfb, yfb, zfb)
    print('Shape: ' + str(type(fgrid)))
    filt = fourier_filter(fgrid, length_scale)
    field['c'] *= filt

    # dom = field.domain
    # slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
    # coeff = []
    #
    # for i in range(dom.dim)[::-1]:
    #     coeff.append(np.linspace(0, 1, dom.global_coeff_shape[i], endpoint=False))
    # cc = np.meshgrid(*coeff)
    #
    # field_filter = np.zeros(dom.local_coeff_shape, dtype='bool')
    # # for i in range(dom.dim):
    # #     field_filter = field_filter | (cc[i][slice])
    #
    # print('Type: ' + str(type(field['g'])) + '\n')
    # print('Shape: ' + str(field['g'].shape))


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
model = dns.DNS_3P_Box(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, ν=1.0, closure=closure)
# model = channel_mod.ChannelMod_DNS(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, xleft=nx, yleft=ny, zbottom=nz,
# 								   nu=1.0, rho=1.0)
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
cadence = 10

# flow = flow_tools.GlobalFlowProperty(model.solver, cadence=cadence)
# flow.add_property("sqrt(u*u + v*v + w*w) / ν", name='Re')


# def average_Re(model): return flow.volume_average('Re')


# def max_Re(model): return flow.max('Re')


# model.add_log_tasks(avg_Re=average_Re, max_Re=max_Re)
model.stop_at(sim_time=np.inf, wall_time=np.inf, iteration=100)

print('Elapsed time: ' + str((time.time() - startt)))

startt = time.time()

# Run the simulation, plot the pressure field occasionally
while model.solver.ok:
    model.solver.step(dt)
    filter_field(model.u, 1)
    filter_field(model.v, 1)
    filter_field(model.w, 1)
    filter_field(model.p, 1)

# plot_bot_3d(model.solver.state['p'], 1, 1, func=log_magnitude)
# plt.savefig('img/dns_' + str(model.solver.iteration / 10) + '.png')
print('Elapsed time: ' + str((time.time() - startt)))
