import sys;

sys.path.append('../dedaLES')

import numpy as np

from numpy import pi
from dedalus import public as de

from dedaLES.dedaLES.flows import Flow
from dedaLES.dedaLES.utils import add_parameters, bind_parameters, add_first_derivative_substitutions
from dedaLES.dedaLES.closures import add_closure_substitutions, add_closure_variables, add_closure_equations


class NavierStokesTriplyPeriodicFlow(Flow):
	'''
	Flow in a triply-periodic box with optional rotation.
	'''

	def __init__(self, nx=32, ny=32, nz=32, Lx=2 * pi, Ly=2 * pi, Lz=2 * pi, ν=1.05e-6, u_bg="0", v_bg="0", w_bg="0",
	             p_bg="0", include_linear_bg=False, closure=None, **params):
		'''
		Create a representation of a triply-periodic box with optional rotation.

		:param nx: (int) Grid resolution in :math:`x`
		:param ny: (int) Grid resolution in :math:`y`
		:param nz: (int) Grid resolution in :math:`z`
		:param Lx: (float) Domain extent in :math:`x`
		:param Ly: (float) Domain extent in :math:`y`
		:param Lz: (float) Domain extent in :math:`z`
		:param ν: (float) 'Molecular' viscosity
		:param u_bg:
		:param v_bg:
		:param w_bg:
		:param p_bg:
		:param include_linear_bg:
		:param closure: (None or closure.EddyViscosityClosure) Turbulent closure for Large Eddy Simulation
		:param params: Additional parameters to be added to the dedalus problem.
		'''
		Flow.__init__(self, nx, ny, nz, Lx, Ly, Lz)

		# Set up boundaries
		self.xlimits = (-Lx / 2, Lx / 2)
		self.ylimits = (-Ly / 2, Ly / 2)
		self.zlimits = (-Lz / 2, Lz / 2)

		# Create bases and domain
		self.xbasis = xbasis = de.Fourier('x', nx, interval=self.xlimits, dealias=3 / 2)
		self.ybasis = ybasis = de.Fourier('y', ny, interval=self.ylimits, dealias=3 / 2)
		self.zbasis = zbasis = de.Fourier('z', nz, interval=self.zlimits, dealias=3 / 2)
		self.domain = domain = de.Domain([xbasis, ybasis, zbasis], grid_dtype=np.float64)
		self.x = domain.grid(0)
		self.y = domain.grid(1)
		self.z = domain.grid(2)

		# Initialize dedalus solver
		bind_parameters(self, ν=ν, **params)
		variables = ['p', 'u', 'v', 'w']
		add_closure_variables(variables, closure)
		self.problem = problem = de.IVP(domain, variables=variables, time='t')
		add_parameters(problem, ν=ν, **params)
		bind_parameters(self, ν=ν, **params)

		add_first_derivative_substitutions(self.problem,
		                                   ['p', 'u', 'v', 'w'],
		                                   ['x', 'y', 'z'])
		add_first_derivative_substitutions(self.problem,
		                                   ['ux', 'vx', 'wx', 'uy', 'vy', 'wy', 'uz', 'vz', 'wz'],
		                                   ['x', 'y', 'z'])

		# R: Reynolds number
		# b: Density
		# m: mu
		problem.substitutions['R'] = '1.0'
		problem.substitutions['b'] = '1.0'
		problem.substitutions['m'] = '1.0'

		# TODO separate linear and background terms

		# Add momentum equations
		matder_x = 'b * (u * ux + v * uy + w * uz + dt(u))'  # material derivative terms
		matder_y = 'b * (u * vx + v * vy + w * vz + dt(v))'
		matder_z = 'b * (u * wx + v * wy + w * wz + dt(w))'
		lapterm_x = 'm * (uxx + uyy + uzz)'  # laplacian terms
		lapterm_y = 'm * (vxx + vyy + vzz)'
		lapterm_z = 'm * (wxx + wyy + wzz)'
		graddiv_x = '1.0/3.0 * m * (uxx + vxy + wxz)'  # grad div terms
		graddiv_y = '1.0/3.0 * m * (uxy + vyy + wyz)'
		graddiv_z = '1.0/3.0 * m * (uxz + vyz + wzz)'
		mom_x = f'{matder_x} = -px + {lapterm_x} + {graddiv_x}'  # final momentum terms
		mom_y = f'{matder_y} = -py + {lapterm_y} + {graddiv_y}'
		mom_z = f'{matder_z} = -pz + {lapterm_z} + {graddiv_z}'
		problem.add_equation(mom_x)  # add the equations to the solver
		problem.add_equation(mom_y)
		problem.add_equation(mom_z)

		# Continuity equation
		problem.add_equation("p = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

	def build_solver(self, timestepper='RK443'):
		'''
		Build a NavierStokesTriplyPeriodicFlow solver.

		:param timestepper: (str) timestepper type for dedalus solver
		:return: dedalus solver for this flow problem
		'''
		Flow.build_solver(self, timestepper=timestepper)

		self.u = self.solver.state['u']
		self.v = self.solver.state['v']
		self.w = self.solver.state['w']
