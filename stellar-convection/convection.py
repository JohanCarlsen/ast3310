import numpy as np 
import FVis3 as FVis
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.constants as const 
from matplotlib.animation import FuncAnimation, PillowWriter
from random import randint
from time import perf_counter

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
		self.gauss = np.zeros((self.N_x, self.N_y))

		# Variables
		self.T = np.zeros((self.N_x, self.N_y)) 		# Temperature [K]
		self.P = np.zeros((self.N_x, self.N_y))			# Pressure [Pa]
		self.u = np.zeros((self.N_x, self.N_y))			# Horizontal velocity component [m/s]
		self.w = np.zeros((self.N_x, self.N_y))			# Vertical velocity component [m/s]
		self.rho = np.zeros((self.N_x, self.N_y)) 		# Density [kg/m^3]
		self.e_int = np.zeros((self.N_x, self.N_y)) 	# Internal energy [J/m^3]

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

			self.T[:, i-1] = self.T[:, i] - dT * self.dy + self.gauss[:, i-1]
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
		self.e_int[:, -1] = - self.dy * self.G * self.rho[:, -1] / self.R_sun**2 \
							+ 1/3 * (4 * self.e_int[:, -2] - self.e_int[:, -3])

		self.e_int[:, 0] = - self.dy * self.G * self.rho[:, 0] / self.R_bot**2 \
							+ 1/3 * (4 * self.e_int[:, 1] - self.e_int[:, 2])

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

		self.dRhoWdt = - rhoW * (dudx + dwdy) - self.u * dRhoWdx - self.w * dRhoWdy - dPdy + self.rho * self.g

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

		if dt < 0.01:

			self.dt = 0.01

		else: 

			self.dt = dt

	def central_x(self, variable):
		'''
		Central difference scheme in x direction.
		'''
		phi = variable

		phi_before = np.roll(phi, -1, axis=0)
		phi_after = np.roll(phi, 1, axis=0)

		dphidx = (phi_after - phi_before) / (2 * self.dx)

		return dphidx

	def central_y(self, variable):
		'''
		Central difference scheme in y direction.
		'''
		phi = variable

		phi_before = np.roll(phi, 1, axis=-1)
		phi_after = np.roll(phi, -1, axis=-1)

		dphidy = (phi_after - phi_before) / (2 * self.dy)

		return dphidy 

	def upwind_x(self, variable, velocity_comp):
		'''
		Upwind difference scheme in x direction.
		'''
		phi = variable
		v = velocity_comp

		dphidx = np.zeros((self.N_x, self.N_y))

		phi_before = np.roll(phi, -1, axis=0)
		phi_after = np.roll(phi, 1, axis=0)

		dphidx[v >= 0] = (phi[v >= 0] - phi_after[v >= 0]) / self.dx 
		dphidx[v < 0] = (phi_before[v < 0] - phi[v < 0]) / self.dx

		return dphidx

	def upwind_y(self, variable, velocity_comp):
		'''
		Upwind difference scheme in y direction.
		'''
		phi = variable
		v = velocity_comp

		dphidy = np.zeros((self.N_x, self.N_y))

		phi_before = np.roll(phi, -1, axis=-1)
		phi_after = np.roll(phi, 1, axis=-1)

		dphidy[v >= 0] = (phi[v >= 0] - phi_before[v >= 0]) / self.dy 
		dphidy[v < 0] = (phi_after[v < 0] - phi[v < 0]) / self.dy

		return dphidy

	def hydro_solver(self):
		'''
		Solver of the hydrodynamic equations
		'''
		self.boundary_condition()	# Set boundary conditions
		self.timestep()				# Update timestep 

		e_new = self.e_int + self.dedt * self.dt
		rho_new = self.rho + self.drhodt * self.dt 
		u_new = (self.rho * self.u + self.dRhoUdt * self.dt) / rho_new 
		w_new = (self.rho * self.w + self.dRhoWdt * self.dt) / rho_new

		R = self.R_sun
		M = self.M_sun

		for i in range(self.N_y - 1, 0, -1):

			dM = 4 * np.pi * R**2 * self.rho[:, i]
			dP = - self.G * M * self.rho[:, i] / R**2
			dT = self.nabla * self.T[:, i] / self.P[:, i] * dP

			self.T[:, i-1] = self.T[:, i] - dT * self.dy 
			self.P[:, i-1] = self.P[:, i] - dP * self.dy 

			M -= dM 
			R -= self.dy

		self.e_int[:], self.rho[:], self.u[:], self.w[:] = e_new, rho_new, u_new, w_new
		self.T[:], self.P[:] = self.T, self.P

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

	def create_gaussian_pertubation(self, amplitude=0.05, x_0=150, y_0=50, stdev_x=25, stdev_y=25):
		'''
		Create a 2D gaussian. Default amplitude is 5 percent of initial temperature,
		and the standard devaiations are both 25. The center of the gaussian will be
		in the center of the box. This will add the gaussian pertubation to the initial temperature.
		'''
		A, sigma_x, sigma_y = amplitude * 5778, stdev_x, stdev_y
		
		for i in range(self.N_x):

			for j in range(self.N_y):

				x, y = i, j

				gauss_ij = A * np.exp(-((x - x_0)**2 / (2 * sigma_x**2) \
									  + (y - y_0)**2 / (2 * sigma_y**2)))

				self.gauss[i, j] = gauss_ij

test = Convection2D()
test.create_gaussian_pertubation()
test.initialise()

total_time = 0
end_time = 60
time_list = [0]
u_list = [test.u]
w_list = [test.w]
T_list = [test.T]
P_list = [test.P]
rho_list = [test.rho]
e_list = [test.e_int]

i = 0

# start = perf_counter()

# while time_list[i] <= end_time:

# 	dt = test.hydro_solver()

# 	total_time += dt 
# 	time_list.append(total_time)
# 	u_list.append(test.u)
# 	w_list.append(test.w)
# 	T_list.append(test.T)
# 	P_list.append(test.P)
# 	rho_list.append(test.rho)
# 	e_list.append(test.e_int)

# 	i += 1

# stop = perf_counter()
# comp_time = stop - start 
# print(f'Total computation time for {end_time} s in {i-1} iterations: {comp_time//60:2.0f} min {comp_time%60:4.1f} s.')

# t = np.array(time_list)
# u = np.array(u_list).T
# w = np.array(w_list).T
# e = np.array(e_list).T
# T = np.array(T_list).T
# P = np.array(P_list).T 
# rho = np.array(rho_list).T

# fig, ax = plt.subplots(figsize=(12.8, 7.2))

# ax.set_title(f'Time: t = {t[0]:4.1f} s')
# im = ax.imshow(u[:, :, 0], origin='lower', norm=plt.Normalize(np.min(u), np.max(u)), cmap='plasma', animated=True)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='3%', pad=0.04)
# fig.colorbar(im, cax=cax)

# def func(i):

# 	ax.clear()
# 	ax.set_title(f'Time: t = {t[i]:4.1f} s')
# 	ax.imshow(u[:, :, i], origin='lower', norm=plt.Normalize(np.min(u), np.max(u)), cmap='plasma', animated=True)
# 	# im.set_array(u[:, :, i])

# start = perf_counter()
# ani = FuncAnimation(fig, func, interval=1)
# # ani.save('test.gif', writer=PillowWriter(fps=144, bitrate=3400), dpi=100)
# stop = perf_counter()
# anim_time = stop - start 
# # print(f'It took {anim_time//60:2.0f} min {anim_time%60:4.1f} s to create and save the animation.')

# plt.show()


# ax.set_title('i=0, t=0')
# im = ax.contourf(test.u.T, levels=100, cmap='plasma')#, vmin=0, vmax=3e6)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# cbar = fig.colorbar(im, cax=cax)
# cbar.ax.invert_yaxis()

# def animate(i):

# 	ax.clear()
# 	dt = test.hydro_solver()
# 	time = (i+1) * dt
# 	ax.set_title(f'i={i}, t={time:5.2f} s')
# 	ax.contourf(test.u.T, levels=100, cmap='plasma')#, vmin=0, vmax=3e6)

# ani = FuncAnimation(fig, animate, frames=9000)
# ani.save('test.gif', writer=PillowWriter(fps=50))

vis = FVis.FluidVisualiser()
# vis.save_data(80, test.hydro_solver, rho=test.rho.T, u=test.u.T, \
# 										w=test.w.T, e=test.e_int.T, \
# 										P=test.P.T, T=test.T.T)

# vis.animate_2D('w', video_name='w-80-secs', height=4.6, cmap='plasma', folder='FVis_output_80_seconds_simulation', save=True)
# vis.animate_2D('u', video_name='u-80-secs', height=4.6, cmap='plasma', folder='FVis_output_80_seconds_simulation', save=True)
# vis.animate_2D('T', video_name='T-80-secs', height=4.6, cmap='plasma', folder='FVis_output_80_seconds_simulation', save=True)
# vis.animate_2D('P', video_name='P-80-secs', height=4.6, cmap='plasma', folder='FVis_output_80_seconds_simulation', save=True)
# vis.animate_2D('e', video_name='e-80-secs', height=4.6, cmap='plasma', folder='FVis_output_80_seconds_simulation', save=True)
# vis.animate_2D('rho', video_name='rho-80-secs', height=4.6, cmap='plasma', folder='FVis_output_80_seconds_simulation', save=True)
# vis.plot_avg('rho', folder='FVis_output_2023-05-17_10-49')

# fig, ax = plt.subplots(figsize=(10, 4.6))
# im = ax.imshow(test.T.T, cmap='plasma')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=.05)
# cbar = fig.colorbar(im, cax=cax)
# cbar.ax.invert_yaxis()
# ax.invert_yaxis()

# plt.show()





