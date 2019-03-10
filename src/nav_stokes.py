def add_vanilla_vars():
	'''
	Construct the variables we need for compressible flow Navier-Stokes equations, including
	the first and second partials of {u, v, w}.
	'''
	variables = ['p']
	for var in ['u', 'v', 'w']:
		variables.append(var)
		for coord in ['x', 'y', 'z']:
			variables.append(f'{var}{coord}')
			for coord2 in ['x', 'y', 'z']:
				if coord <= coord2:	# don't repeat
					variables.append(f'{var}{coord}{coord2}')

	return variables

def reduce_with_system(problem, v2r=['u', 'v', 'w'], c2r=['x','y','z'], depth=2):
	'''
	Add the system of equations to reduce a higher order diffeq to a system
	of lower order diffeqs, i.e. ux = dx(u).
	'''
	for var in v2r:
		for coord in c2r:
			problem.add_equation(f'{var}{coord} - d{coord}({var}) = 0')
			for coord2 in c2r:
				if coord <= coord2:	# don't repeat
					problem.add_equation(f'{var}{coord}{coord2} = d{coord2}({var}{coord})')


def add_vanilla_eqs(problem):
	'''
	Add the compressible flow Navier-Stokes equations, including adding the parameters
	from the params dict.
	'''
	# \mu \nabla^2 \mathbf{u} terms
	lapterm_x = 'm * (uxx + uyy + uzz)'
	lapterm_y = 'm * (vxx + vyy + vzz)'
	lapterm_z = 'm * (wxx + wyy + wzz)'

	# \frac{\mu}{3} \nabla \nabla \cdot \mathbf{u} terms
	graddiv_x = '1.0/3.0 * m * (uxx + vxy + wxz)'
	graddiv_y = '1.0/3.0 * m * (uxy + vyy + wyz)'
	graddiv_z = '1.0/3.0 * m * (uxz + vyz + wzz)'

	# \rho \mathbf{u} \cdot \nabla \mathbf{u} terms
	dotdel_x = '1 * (u * ux + v * uy + w * uz)'
	dotdel_y = '1 * (u * vx + v * vy + w * vz)'
	dotdel_z = '1 * (u * wx + v * wy + w * wz)'

	mom_x = f'1 * dt(u) + dx(p) = {lapterm_x} + {graddiv_x} - {dotdel_x}'
	mom_y = f'1 * dt(v) + dy(p) = {lapterm_y} + {graddiv_y} - {dotdel_y}'
	mom_z = f'1 * dt(w) + dz(p) = {lapterm_z} + {graddiv_z} - {dotdel_z}'

	problem.add_equation(mom_x)
	problem.add_equation(mom_y)
	problem.add_equation(mom_z)

	# Continuity equation
	problem.add_equation('ux + vy + wz = 0', condition='(nx != 0) or (ny != 0) or (nz != 0)')
	problem.add_equation('p = 0', condition='(nx == 0) and (ny == 0) and (nz == 0)')
