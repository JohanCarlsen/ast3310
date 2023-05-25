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
		self.g = self.G * self.M_sun / self.R_sun**2 	# [m/s^2]

		# Courant-Friedrichs-Lewy condition constant
		self.p = 0.1

		# Gaussian pertubation
		self.gauss = np.zeros((self.N_y, self.N_x))

		# Variables
		self.T = np.zeros((self.N_y, self.N_x)) 		# Temperature [K]
		self.P = np.zeros((self.N_y, self.N_x))			# Pressure [Pa]
		self.u = np.zeros((self.N_y, self.N_x))			# Horizontal velocity component [m/s]
		self.w = np.zeros((self.N_y, self.N_x))			# Vertical velocity component [m/s]
		self.rho = np.zeros((self.N_y, self.N_x)) 		# Density [kg/m^3]
		self.e_int = np.zeros((self.N_y, self.N_x)) 	# Internal energy [J/m^3]

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

		self.nabla = 0.5 	# dlnT/(dlnP) has to be slightly larger than 2/5

		self.T[-1, :] = self.T_top + self.gauss[-1, :]
		self.P[-1, :] = self.P_top
		self.e_int[-1, :] = e_top
		self.rho[-1, :] = rho_top 

		R = self.R_sun
		M = self.M_sun

		for i in range(self.N_y - 1, 0, -1):

			dM = 4 * np.pi * R**2 * self.rho[i, :]
			dP = - self.g * self.rho[i, :]
			dT = self.nabla  * self.T[i, :] / self.P[i, :] * dP

			self.T[i-1, :] = self.T[i, :] - dT * self.dy + self.gauss[i-1, :]
			self.P[i-1, :] = self.P[i, :] - dP * self.dy 

			self.e_int[i-1, :] = 3 * self.P[i-1, :] / 2 
			self.rho[i-1, :] = 2 * self.e_int[i-1, :] / 3 * self.mu * self.m_u / (self.k_b * self.T[i-1, :])

	def boundary_condition(self):
		'''
		Vertical boundary conditions for energy, density, and velocity.
		'''
		# Vertical boundary for vertcial velocity component
		self.w[0, :] = 0
		self.w[-1, :] = 0

		# Vertical boundary for horizontal velocity component
		self.u[0, :] = (- self.u[2, :] + 4 * self.u[1, :]) / 3
		self.u[-1, :] = (- self.u[-3, :] + 4 * self.u[-2, :]) / 3

		# Vertical boundary for energy and density
		self.e_int[-1, :] = (4 * self.e_int[-2, :] - self.e_int[-3, :]) \
						  / (3 - 2 * self.dy * self.g * self.mu * self.m_u \
						  / (self.k_b * self.T[-1, :]))

		self.e_int[0, :] = (4 * self.e_int[1, :] - self.e_int[2, :]) \
						  / (3 - 2 * self.dy * self.g * self.mu * self.m_u \
						  / (self.k_b * self.T[0, :]))

		self.rho[-1, :] = 2 * self.e_int[-1, :] / 3 * self.mu * self.m_u / (self.k_b * self.T[-1, :])
		self.rho[0, :] = 2 * self.e_int[0, :] / 3 * self.mu * self.m_u / (self.k_b * self.T[0, :])

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
		dRhoWdx = self.upwind_x(rhoW, self.u)
		dRhoWdy = self.upwind_y(rhoW, self.w)
		dPdy = self.central_y(self.P)

		self.dRhoWdt = - rhoW * (dudx + dwdy) - self.u * dRhoWdx - self.w * dRhoWdy - dPdy - self.rho * self.g

		# Energy equation 
		dudx = self.central_x(self.u)
		dwdy = self.central_y(self.w)
		dedx = self.upwind_x(self.e_int, self.u)
		dedy = self.upwind_y(self.e_int, self.w)

		self.dedt = - (self.e_int + self.P) * (dudx + dwdy) - self.u * dedx - self.w * dedy

		# Making sure there are no zero-divisions 
		non_zero_u = self.u != 0
		non_zero_w = self.w != 0

		rho, e, u, w = self.rho.copy(), self.e_int.copy(), self.u.copy(), self.w.copy()

		rel_rhoU = np.abs(self.dRhoUdt[non_zero_u] / rhoU[non_zero_u])
		rel_rhoW = np.abs(self.dRhoWdt[non_zero_w] / rhoW[non_zero_w])

		if len(rel_rhoU) == 0 or len(rel_rhoW) == 0:

			rel_rhoU = 0 
			rel_rhoW = 0

		rel_rho = np.abs(self.drhodt / rho)
		rel_e = np.abs(self.dedt / e)
		rel_x = np.abs(self.u / self.dx)
		rel_y = np.abs(self.w / self.dy)

		max_vals = np.array([np.max(rel_rho), np.max(rel_rhoU), np.max(rel_rhoW), \
							 np.max(rel_e), np.max(rel_x), np.max(rel_y)])

		delta = np.max(max_vals)

		if delta == 0:

			delta = 1
		
		dt = self.p / delta 

		if dt < 0.1:

			self.dt = 0.1

		else: 

			self.dt = dt

	def central_x(self, variable):
		'''
		Central difference scheme in x direction.
		'''
		phi = variable

		phi_prev = np.roll(phi, 1, axis=-1)
		phi_next = np.roll(phi, -1, axis=-1)

		dphidx = (phi_next - phi_prev) / (2 * self.dx)

		return dphidx

	def central_y(self, variable):
		'''
		Central difference scheme in y direction.
		'''
		phi = variable

		phi_prev = np.roll(phi, 1, axis=0)
		phi_next = np.roll(phi, -1, axis=0)

		dphidy = (phi_next - phi_prev) / (2 * self.dy)

		return dphidy 

	def upwind_x(self, variable, velocity_comp):
		'''
		Upwind difference scheme in x direction.
		'''
		phi = variable
		v = velocity_comp

		# dphidx = np.zeros((self.N_y, self.N_x))

		phi_prev = np.roll(phi, 1, axis=-1)
		phi_next = np.roll(phi, -1, axis=-1)

		pos_vel = (phi - phi_prev) / self.dx
		neg_vel = (phi_next - phi) / self.dx

		dphidx = np.where(v < 0, neg_vel, pos_vel)

		# dphidx[v >= 0] = (phi[v >= 0] - phi_prev[v >= 0]) / self.dx 
		# dphidx[v < 0] = (phi_next[v < 0] - phi[v < 0]) / self.dx

		return dphidx

	def upwind_y(self, variable, velocity_comp):
		'''
		Upwind difference scheme in y direction.
		'''
		phi = variable
		v = velocity_comp

		# dphidy = np.zeros((self.N_y, self.N_x))

		phi_prev = np.roll(phi, 1, axis=0)
		phi_next = np.roll(phi, -1, axis=0)

		pos_vel = (phi - phi_prev) / self.dy 
		neg_vel = (phi_next - phi) / self.dy 

		dphidy = np.where(v < 0, neg_vel, pos_vel)

		# dphidy[v >= 0] = (phi[v >= 0] - phi_prev[v >= 0]) / self.dy 
		# dphidy[v < 0] = (phi_next[v < 0] - phi[v < 0]) / self.dy

		return dphidy

	def hydro_solver(self):
		'''
		Solver of the hydrodynamic equations
		'''
		self.timestep()				# Update timestep 

		e_new = self.e_int + self.dedt * self.dt
		rho_new = self.rho + self.drhodt * self.dt 
		u_new = (self.rho * self.u + self.dRhoUdt * self.dt) / rho_new 
		w_new = (self.rho * self.w + self.dRhoWdt * self.dt) / rho_new

		self.e_int[:], self.rho[:], self.u[:], self.w[:] = e_new, rho_new, u_new, w_new

		self.boundary_condition()	# Set boundary conditions

		self.P[:] = 2 * self.e_int / 3
		self.T[:] = self.P * self.mu * self.m_u / (self.rho * self.k_b)

		return self.dt

	def sanity_check(self):
		'''
		Weather to run a sanity check. If called, random
		values will	be inserted in the arrays for u and w.
		'''
		dt = self.hydro_solver()
		num = randint(1, 3)
		self.w[:] = num 
		self.u[:] = num 

		return dt

	def create_gaussian_pertubation(self, amplitude=0.1, x_0=150, y_0=80, stdev_x=20, stdev_y=20):
		'''
		Create a 2D gaussian. Default amplitude is 10 percent of initial temperature,
		and the standard devaiations are both 20. The center of the gaussian will be
		in the upper center of the box. This will add the gaussian pertubation to the initial temperature.
		'''
		A, sigma_x, sigma_y = amplitude * 5778, stdev_x, stdev_y
		
		for i in range(self.N_y):

			for j in range(self.N_x):

				y, x = i, j

				gauss_ij = A * np.exp(-((x - x_0)**2 / (2 * sigma_x**2) \
									  + (y - y_0)**2 / (2 * sigma_y**2)))

				self.gauss[i, j] = gauss_ij

	def plot_parameter(self, parameter, save=False, fig_name=None, title=None):

		if save is True and fig_name is None:

			print('When saving figures, please enter the figure file name.')
			exit()

		p = parameter

		fig, ax = plt.subplots(figsize=(10, 5))

		if title is not None:

			ax.set_title(title)

		im = ax.imshow(p, cmap='jet', origin='lower', norm=plt.Normalize(np.min(p), np.max(p)))
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		cbar = fig.colorbar(im, cax=cax)
		cbar.ax.invert_yaxis()

		if save:

			filename = 'figures/' + fig_name

			fig.savefig(filename + '.pdf')
			fig.savefig(filename + '.png')

if __name__ == '__main__':

	# Setting units to Mm, and the correct extent of the box.
	units = {'Lx': 'Mm', 'Lz':'Mm'}
	extent = [0, 12, 0, 4]

	# Creating and initializing the box for the sanity check.
	sanity_box = Convection2D()
	sanity_box.initialise()

	# Will simulate for 60 seconds, taking snapshots every 10'th second.
	t_sanity = 60 
	snapshots_sanity = list(np.arange(0, 61, 10))

	vis_sanity = FVis.FluidVisualiser()
	vis_sanity.save_data(t_sanity, sanity_box.sanity_check, u=sanity_box.u, w=sanity_box.w, \
						 e=sanity_box.e_int, T=sanity_box.T, P=sanity_box.P, rho=sanity_box.rho)

	vis_sanity.animate_2D('T', height=4.6, quiverscale=0.25, save=True, video_name=f'figures/animations/sanity_T_{t_sanity}-sec', \
						   units=units, extent=extent)

	vis_sanity.animate_2D('T', height=4.6, quiverscale=0.25, snapshots=snapshots_sanity, video_name=f'sanity_T_{t_sanity}-sec', \
						   units=units, extent=extent)

	# Creating the simulation box and adding a gaussian perturbation to the initial temperature.
	sim_box = Convection2D()
	sim_box.create_gaussian_pertubation()
	sim_box.initialise()

	# Simulate for 10 minutes (600 seconds), taking snapshots every minute.
	t_sim = 600 
	snapshots = list(np.arange(0, 601, 60))

	vis = FVis.FluidVisualiser()
	vis.save_data(t_sim, sim_box.hydro_solver, u=sim_box.u, w=sim_box.w, \
				  e=sim_box.e_int, T=sim_box.T, P=sim_box.P, rho=sim_box.rho)

	vis.animate_2D('T', height=4.6, quiverscale=0.25, save=True, video_name=f'figures/animations/T_{t_sim}-sec', \
				    units=units, extent=extent)

	vis.animate_2D('T', height=4.6, quiverscale=0.25, snapshots=snapshots, video_name=f'T_{t_sim}-sec', \
				    units=units, extent=extent)




