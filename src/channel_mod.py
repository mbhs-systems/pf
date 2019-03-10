import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod

# import dedalus as de
from dedalus import public as de

from dedaLES.flows import ChannelFlow
from dedaLES.closures import *
from dedaLES.utils import *

class ChannelMod(ChannelFlow):
    def __init__(self, nx, ny, nz, Lx, Ly, Lz, xleft, yleft, zbottom, **substitutions):
        super().__init__(nx, ny, nz, Lx, Ly, Lz, xleft, yleft, zbottom)

        self.xbasis = xbasis = de.Fourier('x', nx, interval=self.xlimits, dealias=3/2)
        self.ybasis = ybasis = de.Fourier('y', ny, interval=self.ylimits, dealias=3/2)
        self.zbasis = zbasis = de.Fourier('z', nz, interval=self.zlimits, dealias=3/2)
        self.domain = domain = de.Domain([xbasis, ybasis, zbasis], grid_dtype=np.float64)

        # Add variables (u, v, w components of the flow field, their first x,y,z partials, and the pressure field p)
        vrep = ['u', 'v', 'w']
        variables = ['p'] + vrep

        for var in vrep:
            for coord in ['x', 'y', 'z']:
                variables.append(f'{var}{coord}')
                for coord2 in ['x', 'y', 'z']:
                    variables.append(f'{var}{coord}{coord2}')

        print(variables)

        self.problem = de.IVP(domain, variables=variables, time='t')

        for var in vrep:
            for coord in ['x', 'y', 'z']:
                self.problem.add_equation(f'{var}{coord} = d{coord}({var})')
                for coord2 in ['x', 'y', 'z']:
                    self.problem.add_equation(f'{var}{coord}{coord2} = d{coord2}({var}{coord})')

        # add_first_derivative_substitutions(problem, ['u', 'v', 'w', 'b'], ['x', 'y'])
        self._add_substitutions(**substitutions)

        # Get substitution strings for the two terms that differ between DNS, LES, and RANS
        nla = self._get_nonlinear_advection()
        st = self._get_strain_term()

        # Add conservation of momentum equations
        for i in range(3):
            icoord = ['x', 'y', 'z'][i]
            icomp = ['u', 'v', 'w'][i]
            eqstr = f'dt({icomp}) - d{icoord}(p) = {nla[i]} + {st[i]}'
            print(eqstr)
            self.problem.add_equation(eqstr)

        # Add continuity equations
        self.problem.add_equation('ux + vy + wz = 0', condition='(nx != 0) or (ny != 0) or (nz != 0)')
        self.problem.add_equation('p = 0', condition='(nx == 0) and (ny == 0) and (nz == 0)')


    def _add_substitutions(self, **substitutions):
        pass

    def build_solver(self):
        super().build_solver()
        self.u = self.solver.state['u']
        self.v = self.solver.state['v']
        self.w = self.solver.state['w']


    @abstractmethod
    def _get_nonlinear_advection(self):
        pass


    @abstractmethod
    def _get_strain_term(self):
        pass


class ChannelMod_DNS(ChannelMod):
    '''
    Channel flow model with the substitutions for DNS
    '''

    def _add_substitutions(self, **substitutions):
        self.problem.substitutions['nu'] = substitutions['nu']
        self.problem.substitutions['rho'] = substitutions['rho']


    def _get_nonlinear_advection(self):
        nla = []
        for coord in ['u', 'v', 'w']:
            nla.append(f'u * {coord}x + v * {coord}y + w * {coord}z')
        return nla


    def _get_strain_term(self):
        st = []
        for coord in ['u', 'v', 'w']:
            # st.append(f'nu * ({coord}xx + {coord}yy + {coord}zz)')
            st.append(f'({coord}xx + {coord}yy + {coord}zz)')
        return st

# class ChannelMod_LES(ChannelMod):
#     '''
#     Channel flow  model with the substitutions for LES (no filtering here yet)
#     '''
#     def _get_nonlinear_advection(self):
#         nla = []
#         for coord in ['u', 'v', 'w']:
#             nla.append(f'b * (u * {coord}x + v * {coord}y + w * {coord}z)')
#         return nla
