import numpy as np 
import FVis3 as FVis
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.constants as const 
from random import randint

class Convection2D:

	def __init__(self):

		# Box paramters 
		self.x_length = 12e6 	# Box length in x direction [m]
		self.y_length = 4e6 	# Box length in y direction [m]
		self.N_x = 300 			# Points in the x direction of the box 
		self.N_y = 100 			# Points in the y diretion of the box 

		# Spatial step lengths
		self.dx = self.x_length / self.N_x 	# [m]
		self.dy = self.y_length / self.N_y 	# [m]

		# Time step length (will be continously updated)
		self.dt = None

		# Constants 
		self.mu = 0.61 					# Mean molecular mass of an ideal gas
		self.m_u = const.u.value 		# Atomic mass unit [kg]
		self.k_b = const.k_B.value 		# Boltzmann constant [J/K]
		self.G = const.G.value			# Grav. const. [m^3/kgs^2]
		self.R_sun = const.R_sun.value 	# Solar radius [m]
		self.M_sun = const.M_sun.value 	# Solar mass [kg]
		self.L_sun = const.L_sun.value 	# Solar luminosity [W]

		# Constant gravitational acceleration in the y direction 
		self.g = np.array([0., - self.G * self.M_sun / self.R_sun**2]) 	# [m/s^2]

		# Courant-Friedrichs-Lewy condition constant
		self.p = 0.1

		# Gaussian pertubation
		self.gauss = np.zeros((self.N_x, self.N_y))

		# Variables
		self.T = np.zeros((self.N_x, self.N_y)) 		# Temperature [K]
		self.P = np.zeros((self.N_x, self.N_y))			# Pressure [Pa]
		self.u = np.zeros((self.N_x, self.N_y))			# Horizontal velocity component [m/s]
		self.w = np.zeros((self.N_x, self.N_y))			# Vertical velocity component [m/s]
		self.rho = np.zeros((self.N_x, self.N_y)) 		# Density [kg/m^3]
		self.e_int = np.zeros((self.N_x, self.N_y)) 	# Internal energy [J/m^3]

		# Defining the box 
		self.box = np.zeros((self.N_x, self.N_y)) 

	def initialise(self):
		'''
		Initialise the temperature, pressure,
		density, and internal energy.
		'''
		# Temperature and pressure at the top of the box 
		self.T_top = 5778 	# [K]
		self.P_top = 1.8e4 	# [Pa]

		e_top = 3 * self.P_top / 2 
		rho_top = 2 * e_top / 3 * self.mu * self.m_u / (self.k_b * self.T_top)

		self.nabla = 0.4001 	# dlnT/(dlnP) has to be slightly larger than 2/5

		self.T[:, -1] = self.T_top
		self.P[:, -1] = self.P_top
		self.e_int[:, -1] = e_top
		self.rho[:, -1] = rho_top 

		R = self.R_sun
		M = self.M_sun

		for i in range(self.N_y - 1, 0, -1):

			dM = 4 * np.pi * R**2 * self.rho[:, i]
			dP = - self.G * M * self.rho[:, i] / R**2
			dT = self.nabla  * self.T[:, i] / self.P[:, i] * dP

			self.T[:, i-1] = self.T[:, i] - dT * self.dy 
			self.P[:, i-1] = self.P[:, i] - dP * self.dy 

			self.e_int[:, i-1] = 3 * self.P[:, i-1] / 2 
			self.rho[:, i-1] = 2 * self.e_int[:, i-1] / 3 * self.mu * self.m_u / (self.k_b * self.T[:, i-1])

			M -= dM 
			R -= self.dy

		self.R_bot = R 
		self.M_bot = M

	def boundary_condition(self):
		'''
		Vertical boundary conditions for energy, density, and velocity.
		'''
		# Vertical boundary for vertcial velocity component
		self.w[:, 0] = 0
		self.w[:, -1] = 0

		# Vertical boundary for horizontal velocity component
		self.u[:, 0] = (- self.u[:, 2] + 4 * self.u[:, 1]) / 3
		self.u[:, -1] = (- self.u[:, -3] + 4 * self.u[:, -2]) / 3

		# Vertical boundary for energy and density
		self.e_int[:, -1] = 3 * self.P[:, -1] / 2
		self.e_int[:, 0] = 3 * self.P[:, 0] / 2
		self.rho[:, -1] = 2 * self.e_int[:, -1] / 3 * self.mu * self.m_u / (self.k_b * self.T[:, -1])
		self.rho[:, 0] = 2 * self.e_int[:, 0] / 3 * self.mu * self.m_u / (self.k_b * self.T[:, 0])

	def timestep(self):
		'''
		Calculate the timestep by evaluating the spatial derivatives 
		of all the variables, and then choose the optimal timestep.
		'''
		# Continuity equation 
		dudx = self.central_x(self.u)
		dwdy = self.central_y(self.w)
		drhodx = self.upwind_x(self.rho, self.u)
		drhody = self.upwind_y(self.rho, self.w)

		self.drhodt = - self.rho * (dudx + dwdy) - self.u * drhodx - self.w * drhody

		# Horizontal component of momentum equation 
		rhoU = self.rho * self.u 
		dudx = self.upwind_x(self.u, self.u)
		dwdy = self.central_y(self.w)
		dRhoUdx = self.upwind_x(rhoU, self.u)
		dRhoUdy = self.upwind_y(rhoU, self.w)
		dPdx = self.central_x(self.P)

		self.dRhoUdt = - rhoU * (dudx + dwdy) - self.u * dRhoUdx - self.w * dRhoUdy - dPdx

		# Vertical component of momentum equation 
		rhoW = self.rho * self.w 
		dudx = self.central_x(self.u)
		dwdy = self.upwind_y(self.w, self.w)
		dRhoWdx = self.upwind_x(rhoW, self.w)
		dRhoWdy = self.upwind_y(rhoW, self.u)
		dPdy = self.central_y(self.P)

		self.dRhoWdt = - rhoW * (dudx + dwdy) - self.u * dRhoWdx - self.w * dRhoWdy - dPdy + self.rho * self.g[1]

		# Energy equation 
		dudx = self.central_x(self.u)
		dwdy = self.central_y(self.w)
		dedx = self.upwind_x(self.e_int, self.u)
		dedy = self.upwind_y(self.e_int, self.w)

		self.dedt = - (self.e_int + self.P) * (dudx + dwdy) - self.u * dedx - self.w * dedy

		self.boundary_condition()

		rho, e, u, w = self.rho, self.e_int, self.u, self.w

		# Making sure there are no zero-divisions 
		rho[(rho == 0)] = 1
		e[(e == 0)] = 1
		rhoU[(rhoU == 0)] = 1
		rhoW[(rhoW == 0)] = 1

		rho[(abs(rho) > 5)] = 1
		rhoU[(abs(rhoU) > 5)] = 1
		rhoW[(abs(rhoW) > 5)] = 1
		e[(abs(e) > 5)] = 1
		u[(abs(u) > 5)] = 1
		w[(abs(w) > 5)] = 1

		rel_rho = np.abs(self.drhodt / rho)
		rel_rhoU = np.abs(self.dRhoUdt / rhoU)
		rel_rhoW = np.abs(self.dRhoWdt / rhoW)
		rel_e = np.abs(self.dedt / e)
		rel_x = np.abs(self.u / self.dx)
		rel_y = np.abs(self.w / self.dy)

		max_vals = np.array([np.max(rel_rho), np.max(rel_rhoU), np.max(rel_rhoW), \
							 np.max(rel_e), np.max(rel_x), np.max(rel_y)])

		delta = np.max(max_vals)
		# self.dt = self.p / delta
		# self.dt = 0.5
		dt = self.p / delta 

		if dt < 0.5:

			self.dt = 0.5

		else: 

			self.dt = dt

	def central_x(self, variable):
		'''
		Central difference scheme in x direction.
		'''
		Nx = self.N_x 
		phi = variable
		dphidx = np.zeros((self.N_x, self.N_y))

		for i in range(Nx):

			for j in range(1, self.N_y - 2):

				dphi_ij = (phi[(i+1) % Nx, j] - phi[(i + Nx - 1) % Nx, j]) / (2 * self.dx)
				dphidx[i, j] = dphi_ij

		return dphidx

	def central_y(self, variable):
		'''
		Central difference scheme in y direction.
		'''
		phi = variable
		dphidy = np.zeros((self.N_x, self.N_y))

		for i in range(self.N_x):

			for j in range(1, self.N_y - 2):

				dphi_ij = (phi[i, j+1] - phi[i, j-1]) / (2 * self.dy)
				dphidy[i, j] = dphi_ij

		return dphidy 

	def upwind_x(self, variable, velocity_comp):
		'''
		Upwind difference scheme in x direction.
		'''
		Nx = self.N_x 
		phi = variable
		v = velocity_comp
		dphidx = np.zeros((self.N_x, self.N_y))

		for i in range(Nx):

			for j in range(1, self.N_y - 2):

				if v[i, j] >= 0:

					dphi_ij = (phi[i, j] - phi[(i + Nx - 1) % Nx, j]) / self.dx 

				else: 

					dphi_ij = (phi[(i+1) % Nx, j] - phi[i, j]) / self.dx 

				dphidx[i, j] = dphi_ij

		return dphidx

	def upwind_y(self, variable, velocity_comp):
		'''
		Upwind difference scheme in y direction.
		'''
		phi = variable
		v = velocity_comp
		dphidy = np.zeros((self.N_x, self.N_y))

		for i in range(self.N_x):

			for j in range(1, self.N_y - 2):

				if v[i, j] >= 0:

					dphi_ij = (phi[i, j] - phi[i, j-1]) / self.dy 

				else:

					dphi_ij = (phi[i, j+1] - phi[i, j]) / self.dy 

				dphidy[i, j] = dphi_ij

		return dphidy

	def hydro_solver(self):
		'''
		Solver of the hydrodynamic equations
		'''
		self.timestep()		# Update timestep 

		e_new = self.e_int + self.dedt * self.dt
		rho_new = self.rho + self.drhodt * self.dt 
		u_new = (self.rho * self.u + self.dRhoUdt * self.dt) / rho_new 
		w_new = (self.rho * self.w + self.dRhoWdt * self.dt) / rho_new

		self.T[:, -1] = self.T_top
		self.P[:, -1] = self.P_top

		R = self.R_sun
		M = self.M_sun

		for i in range(self.N_y - 1, 0, -1):

			dM = 4 * np.pi * R**2 * self.rho[:, i]
			dP = - self.G * M * self.rho[:, i] / R**2
			dT = self.nabla  * self.T[:, i] / self.P[:, i] * dP

			self.T[:, i-1] = self.T[:, i] - dT * self.dy 
			self.P[:, i-1] = self.P[:, i] - dP * self.dy 

			M -= dM 
			R -= self.dy

		self.e_int[:], self.rho[:], self.u[:], self.w[:] = e_new, rho_new, u_new, w_new

		return self.dt

	def sanity_check(self):
		'''
		Weather to run a sanity check. If called, random
		values will	be inserted in the arrays for u and w.
		'''
		dt = self.hydro_solver()
		num = randint(1, 10)
		self.w[:] = num 
		self.u[:] = num 

		return dt

	def add_gaussian_pertubation(self, amplitude=5e4, x_0=150, y_0=50, stdev_x=50, stdev_y=50):
		'''
		Create a 2D gaussian. Default amplitude is 5e4, and the standard devaiations
		are both 50. The center of the gaussian will be in the center of the box.
		This will add the gaussian pertubation to the initial temperature.
		'''
		A, sigma_x, sigma_y = amplitude, stdev_x, stdev_y
		
		for i in range(self.N_x):

			for j in range(self.N_y):

				x, y = i, j

				gauss_ij = A * np.exp(-((x - x_0)**2 / (2 * sigma_x**2) \
									  + (y - y_0)**2 / (2 * sigma_y**2)))

				self.gauss[i, j] = gauss_ij

		self.T += self.gauss


test = Convection2D()
test.initialise()
test.add_gaussian_pertubation()

vis = FVis.FluidVisualiser()
vis.save_data(180, test.hydro_solver, rho=test.rho.T, u=test.u.T, \
										w=test.w.T, e=test.e_int.T, \
										P=test.P.T, T=test.T.T,)

vis.animate_2D('u', height=4.6, cmap='plasma')

# fig, ax = plt.subplots()
# im = ax.imshow(test.T.T, cmap='plasma')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=.05)
# cbar = fig.colorbar(im, cax=cax)
# cbar.ax.invert_yaxis()
# ax.invert_yaxis()

# plt.show()





