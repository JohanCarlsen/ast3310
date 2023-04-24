'''
*** IMPORTANT NOTE***
This program assumes that the user has the program energy_production.py in a separate 
folder called stellar-engines/. If this program is in the folder user/this-program/,
then the file energy_production.py *has* to be in the folder user/stellar-engines/.
'''
import sys
sys.path.append("../stellar-engines")
from energy_production import EnergyProduction
from cross_section import cross_section
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp
import scipy.constants as const 

class Star:

	def __init__(self, init_mass, init_radius, init_pressure, init_luminosity, init_temperature):

		self.m_0 = init_mass				# [kg]
		self.r_0 = init_radius				# [m]
		self.P_0 = init_pressure 			# [Pa]
		self.L_0 = init_luminosity			# [W]
		self.T_0 = init_temperature 		# [K]

		# Lists to hold the evolutions of the parameters 
		self.m = [self.m_0]; self.r = [self.r_0]; self.P = [self.P_0];
		self.L = [self.L_0]; self.T = [self.T_0]; self.rho = [];
		self.nabla_star = []; self.nabla_stable = []
		self.F_rad = []; self.F_con = []; self.epsilon = []
		self.PP1 = []; self.PP2 = []; self.PP3 = []; self.CNO = []

		# Fractional mass abundances 
		self.X = .7 			# Fraction of hydrogen 
		self.Y_He3 = 1e-10		# Fraction of helium-3
		self.Y = .29 			# Fraction of helium-4 
		self.Z_Li7 = 1e-7		# Fraction of lithium-7
		self.Z_Be7 = 1e-7		# Fraction of beryllium-7
		self.Z_N14 = 1e-11		# Fraction of nitrogen-14

		# Mean molecular weight
		self.mu = 1. / (2 * self.X + self.Y_He3 + (3/4 * self.Y) + (4/7 * self.Z_Li7) \
								   + (5/7 * self.Z_Be7) + (4/7 * self.Z_N14))

		# Adiabatic temperature gradient for ideal gas 
		self.nabla_ad = 2/5

		# Heat capacity at constant pressure 
		self.c_P = 5/2 * const.k / (self.mu * const.m_u)

		# Sun parameters
		self.M_sun = 1.989e30 		# [kg]
		self.L_sun = 3.846e26 		# [W]
		self.R_sun = 6.96e8 		# [m]
		self.avg_rho_sun = 1.408e3	# [kgm^-3]

		# Sanity check parameters
		self.sanity_T = 9e5					# [K]
		self.sanity_R = 0.84 * self.R_sun 	# [m]
		self.sanity_M = 0.99 * self.M_sun	# [kg]

		self.init_sanity = np.array([self.sanity_M, self.sanity_R, self.sanity_T])


	def opacity(self, density, temperature, get_warnings=False, sanity_check=False):
		'''
		Reads the file opacity.txt and creates a rectangular spline, which containes
		interpolated values. If the entered density or temperature is outside of the
		range of the file, the returned value will be extrapolated. Set get_warnings
		to true to recieve a warning when this happens.
		'''

		# Extracting data from opacity.txt.
		log_R = np.loadtxt('opacity.txt', usecols=range(1, 20), max_rows=1)		# [gcm^-3 K^-3]
		data = np.loadtxt('opacity.txt', skiprows=2)
		log_T = data[:, 0]														# [K]
		log_K = data[:, 1:]														# [cm^2 g^-1]

		spline = RectBivariateSpline(log_T, log_R, log_K) 

		if sanity_check:

			T = [3.75, 3.755, 3.755, 3.755, 3.755, 3.77, 3.78, 3.795, 3.77, 3.775, 3.780, 3.795, 3.8]
			R = [-6, -5.95, -5.8, -5.7, -5.5, -5.95, -5.95, -5.95, -5.8, -5.75, -5.7, -5.55, -5.5]

			K = spline.ev(T, R)

			print('\nPERFORMING OPACITY SANITY CHECK\n')
			print(f"{'log10(T)':<12}{'log10(R) [cgs]':<20}{'log10(K) [cgs]':<20}{'K [SI]'}")

			for i in range(len(T)):

				Ti = f"{T[i]:.3f}"
				Ri = f"{R[i]:.2f}"
				Ki = f"{K[i]:.2f}"
				Ki_SI = f"{10**K[i] * 1e-1:.2e}"

				print(f"{'':<1}{Ti:<15}{Ri:<20}{Ki:<15}{Ki_SI}")

			print('\nOPACITY SANITY CHECK FINISHED\n')

			return # Returns None, and terminates the function

		rho_cgs = density * 1e-3 
		log_R = np.log10(rho_cgs / (temperature * 1e-6)**3)
		log_T = np.log10(temperature)

		log_K_cgs = spline.ev(log_T, log_R)		# [cgs]
		K = 10**log_K_cgs / 10 					# [SI]

		return K 

	def density_pressure(self, temperature, density=None, pressure=None, get_info=False):
		'''
		Calculate P(rho,T) if density is given, or rho(P,T), if pressure is given.
		The user can get the calculated quantity, if get_info is set to True.
		'''
		if density is None and pressure is None:

			print('\nError: No argument passed to density_pressure!\n')
			exit()

		T = temperature
		P_rad = 4 / (3 * const.c) * const.sigma * T**4

		if density is not None:

			P_gas = density / (self.mu * const.m_u) * const.k * T
			P = P_rad + P_gas

			if get_info:

				print(f'\nComputed pressure from get_density_pressure: {P:.2e} Pa\n')

			return P

		if pressure is not None:

			rho = (pressure - P_rad) * self.mu * const.m_u / (const.k * T)

			if get_info:

				print(f'\nComputed density from get_density_pressure: {rho:.2e} kg/m\u00b3\n')

			return rho

	def evolve_one_step(self, variables, p, sanity_check=False, include_cycles=False):
		'''
		Evolve the parameters for mass, radius, pressure,
		luminosity, and temperature one step length, using
		variable step length controlled by the factor p.
		'''
		if sanity_check:

			rho = 55.9
			kappa = 3.98

			m, r, T = variables

		else:

			m, r, P, L, T = variables

			rho = self.density_pressure(T, pressure=P)
			kappa = self.opacity(rho, T)

		g = const.G * m / r**2
		H_P = const.k * T / (self.mu * const.m_u * g)
		l_m = H_P
		r_p = l_m / 2
		SQd = 2 / r_p

		U = 64 * const.sigma * T**3 / (3 * kappa * rho**2 * self.c_P) * np.sqrt(H_P / g)
		K = U * SQd / l_m

		if sanity_check:

			nabla_stable = 3.26078

		else:

			nabla_stable = 3 * kappa * rho * H_P * L / (64 * const.sigma * T**4 * const.pi * r**2)

		coeffs = [l_m**2/U, 1, K, (self.nabla_ad - nabla_stable)]
		roots = np.roots(coeffs)
		real_root_idx = np.where(roots.imag == 0)[0][0]
		xi = roots[real_root_idx].real

		nabla_star = xi**2 + xi * K + self.nabla_ad

		F_rad = 16 * const.sigma * T**4 / (3 * kappa * rho * H_P) * nabla_star
		F_con = rho * self.c_P * T * np.sqrt(g) * H_P**(-3/2) * (l_m/2)**2 * xi**3

		star = EnergyProduction(T, rho)
		star.run_all_cycles()
		eps = star.get_total_energy()

		if sanity_check:

			v_exact = 65.6 				# [ms^-1]
			U_exact = 5.94e5			# [m^2]
			H_P_exact = 32.4 			# [Mm] 
			xi_exact = 1.173e-3
			nabla_star_exact = 0.4
			F_con_over_exact = 0.88
			F_rad_over_exact = 0.12

			v = np.sqrt(g/H_P) * l_m/2 * xi
			H_P *= 1e-6
			F_con_over = F_con / (F_con + F_rad)
			F_rad_over = F_rad / (F_con + F_rad)

			H_P_err = abs(H_P - H_P_exact) / abs(H_P_exact) * 100
			v_err = abs(v - v_exact) / abs(v_exact) * 100
			xi_err = abs(xi - xi_exact) / abs(xi_exact) * 100
			U_err = abs(U - U_exact) / abs(U_exact) * 100
			nabla_star_err = abs(nabla_star - nabla_star_exact) / abs(nabla_star_exact) * 100
			F_con_over_err = abs(F_con_over - F_con_over_exact) / abs(F_con_over_exact) * 100
			F_rad_over_err = abs(F_rad_over - F_rad_over_exact) / abs(F_rad_over_exact) * 100

			self.opacity(0, 0, sanity_check=True)

			print("PERFORMING CONVECTIVE ZONE SANITY CHECK\n")
			print(f"{'Param.':<18}{'Exact':<10}{'Computed':<12}{'Rel. err.'}")
			print(f"{'H_P':<13}{H_P_exact:10.3f}{H_P:13.3f}{H_P_err:11.2f} %")
			print(f"{'v':<13}{v_exact:10.3f}{v:13.3f}{v_err:11.2f} %")
			print(f"{'Grad_star':<13}{nabla_star_exact:10.3f}{nabla_star:13.3f}{nabla_star_err:11.2f} %")
			print(f"{'F_con/sum_F':<13}{F_con_over_exact:10.3f}{F_con_over:13.3f}{F_con_over_err:11.2f} %")
			print(f"{'F_rad/sum_F':<13}{F_rad_over_exact:10.3f}{F_rad_over:13.3f}{F_rad_over_err:11.2f} %")
			print(f"{'xi':<13}{xi_exact:10.3e}{xi:13.3e}{xi_err:11.2f} %")
			print(f"{'U':<13}{U_exact:10.3e}{U:13.3e}{U_err:11.2f} %")
			print('\nCONVECTIVE ZONE SANITY CHECK FINISHED\n')

			return	# Returns None, and terminates the function.

		dr = 1 / (4 * const.pi * r**2 * rho)
		dP = -const.G * m / (4 * const.pi * r**4)
		dL = eps

		if nabla_stable > self.nabla_ad:

			dT = nabla_star * T/P * dP

		else:

			dT = -3 * kappa * L / (256 * const.pi**2 * const.sigma * r**4 * T**3)
			nabla_star = nabla_stable

		dm_r = r / abs(dr) 
		dm_P = P / abs(dP) 
		dm_L = L / abs(dL)
		dm_T = T / abs(dT) 

		dm_list = np.array([dm_r, dm_P, dm_L, dm_T]) * p
		dm = np.min(dm_list)

		r_new = r - dr * dm 
		P_new = P - dP * dm 
		L_new = L - dL * dm 
		T_new = T - dT * dm
		m_new = m - dm

		if include_cycles:

			PP1, PP2, PP3, CNO = np.sum(star.PP1), np.sum(star.PP2), np.sum(star.PP3), np.sum(star.CNO)

			return m_new, r_new, P_new, L_new, T_new, rho, nabla_stable, nabla_star, F_rad, F_con, eps, PP1, PP2, PP3, CNO

		else:

			return m_new, r_new, P_new, L_new, T_new, rho, nabla_stable, nabla_star, F_rad, F_con, eps

	def integrate_equtations(self, p, output=True, include_cycles=False):
		'''
		Runs evolve_one_step until either of the
		mass, radius, or luminosity reaches zero.
		The argument is p, which controlls the 
		variable step length of the evolution funciton.
		'''
		i = 0 

		while self.m[i] > 0 and self.r[i] > 0 and self.L[i] > 0:

			variables = np.array([self.m[i], self.r[i], self.P[i], self.L[i], self.T[i]])

			if include_cycles:

				m_i, r_i, P_i, L_i, T_i, rho_i, stable_i, star_i, F_rad_i, F_con_i, eps_i, PP1_i, PP2_i, PP3_i, CNO_i = self.evolve_one_step(variables, p, include_cycles=True)

				self.m.append(m_i); self.r.append(r_i); self.P.append(P_i); self.L.append(L_i)
				self.T.append(T_i); self.rho.append(rho_i); self.nabla_stable.append(stable_i)
				self.nabla_star.append(star_i); self.F_rad.append(F_rad_i)
				self.F_con.append(F_con_i); self.epsilon.append(eps_i)	
				self.PP1.append(PP1_i); self.PP2.append(PP2_i); self.PP3.append(PP3_i); self.CNO.append(CNO_i)

			else:

				m_i, r_i, P_i, L_i, T_i, rho_i, stable_i, star_i, F_rad_i, F_con_i, eps_i = self.evolve_one_step(variables, p)

				self.m.append(m_i); self.r.append(r_i); self.P.append(P_i); self.L.append(L_i)
				self.T.append(T_i); self.rho.append(rho_i); self.nabla_stable.append(stable_i)
				self.nabla_star.append(star_i); self.F_rad.append(F_rad_i)
				self.F_con.append(F_con_i); self.epsilon.append(eps_i)

			i += 1

			if abs(self.m[i] - self.m[i-1]) < 1e15:

				if output:

					print('\nWarning: Step length dm converged to zero!')

				break

		if output:
			
			print(f'\nFinal values after {i-1} iterations:')
			print(f'M/M_0: {self.m[-2]/self.m[0]*100:4.1f} %. M = {self.m[-2]:.3e} kg')
			print(f'R/R_0: {self.r[-2]/self.r[0]*100:4.1f} %. R = {self.r[-2]:.3e} m')
			print(f'L/L_0: {self.L[-2]/self.L[0]*100:4.1f} %. L = {self.L[-2]:.3e} W')


	def get_arrays(self, include_cycles=False):
		'''
		Convert the lists to arrays and returns them. Since the 
		function integrate_equations runs until the parameter 
		value is no longer larger than zero, the last element 
		is discarded.
		'''
		m = np.array(self.m[:-1]); r = np.array(self.r[:-1]); P = np.array(self.P[:-1])
		L = np.array(self.L[:-1]); T = np.array(self.T[:-1]); rho = np.array(self.rho)
		star = np.array(self.nabla_star); stable = np.array(self.nabla_stable)
		F_rad = np.array(self.F_rad); F_con = np.array(self.F_con)
		eps = np.array(self.epsilon)

		if include_cycles:

			PP1 = np.array(self.PP1); PP2 = np.array(self.PP2)
			PP3 = np.array(self.PP3); CNO = np.array(self.CNO)

			return m, r, P, L, T, rho, star, stable, F_rad, F_con, eps, PP1, PP2, PP3, CNO

		else:

			return m, r, P, L, T, rho, star, stable, F_rad, F_con, eps

	def sanity_check(self):
		'''
		Runs the evolve_one_step function with the
		initial values for the sanity check.
		'''
		self.evolve_one_step(self.init_sanity, p=0, sanity_check=True)


if __name__ == '__main__':

	plt.rcParams['lines.linewidth'] = 1
	plt.rcParams['font.size'] = 12
	plt.rcParams['text.usetex'] = True
	plt.rcParams['font.family'] = 'Helvetica'

	# Sun parameters 
	L_sun = 3.846e26		# [W]
	M_sun = 1.989e30 		# [kg]
	R_sun = 6.96e8 			# [m]
	avgRho_sun = 1.408e3 	# [kgm^-3]

	# Fractional mass abundances 
	X = .7 				# Fraction of hydrogen 
	Y_He3 = 1e-10		# Fraction of helium-3
	Y = .29 			# Fraction of helium-4 
	Z_Li7 = 1e-7		# Fraction of lithium-7
	Z_Be7 = 1e-7		# Fraction of beryllium-7
	Z_N14 = 1e-11		# Fraction of nitrogen-14

	mu = 1. / (2 * X + Y_He3 + (3/4 * Y) + (4/7 * Z_Li7) + (5/7 * Z_Be7) + (4/7 * Z_N14))	# Mean molecular mass 

	nabla_ad = 2/5 	# Adiabatic temperature gradient for ideal gas

	# Initial parameters 
	rho_0 = 1.42e-7 * avgRho_sun
	T_0 = 5770
	P_G = rho_0 / (mu * const.m_u) * const.k * T_0		# Gas pressure 
	P_rad = 4/3 * const.sigma * T_0**4 / const.c 		# Radiative pressure 
	P_0 = (P_G + P_rad)
	L_0 = L_sun
	M_0 = M_sun
	R_0 = R_sun

	test = Star(M_0, R_0, P_0, L_0, T_0)
	test.sanity_check()
	test.integrate_equtations(p=1e-2)

	m, r, P, L, T, rho, nabla_star, nabla_stable, F_rad, F_con, eps = test.get_arrays()

	iterations = np.linspace(0, len(m), len(m))

	plt.figure(figsize=(10,4))
	plt.plot(iterations, m/np.max(m), ls='solid', color='black', label=r'$M/M_{max}$')
	plt.plot(iterations, r/np.max(r), ls='dashed', color='black', label=r'$R/R_{max}$')
	plt.plot(iterations, L/np.max(L), ls='dashdot', color='black', label=r'$L/L_{max}$')
	plt.plot(iterations, 0.05*np.ones(len(iterations)), ls='dotted', color='black')
	plt.legend()
	plt.xlabel('Iterations')
	plt.tight_layout()
	plt.savefig('figures/sanity/convergence_check.pdf')
	plt.savefig('figures/sanity/convergence_check.png')

	plt.figure()
	plt.plot(r/np.max(r), eps, lw=1, color='black', label=r'$\partial L/\partial m$')
	plt.yscale('log')
	plt.legend()
	plt.xlabel(r'$R/R_{max}$')
	plt.ylabel(r'Js$^{-1}$')
	plt.tight_layout()
	plt.savefig('figures/sanity/epsilon.pdf')
	plt.savefig('figures/sanity/epsilon.png')

	plt.figure(figsize=(10,4))
	plt.plot(r/np.max(r), T/np.max(T), ls='dashed', color='black', label=r'$T/T_{max}$')
	plt.plot(r/np.max(r), P/np.max(P), ls='dotted', color='black', label=r'$P/P_{max}$')
	plt.plot(r/np.max(r), rho/np.max(rho), ls=(0, (3, 5, 1, 5)), color='black', label=r'$\rho/\rho_{max}$')
	plt.legend()
	plt.xlabel(r'$R/R_{max}$')
	plt.tight_layout()
	plt.savefig('figures/sanity/T-P-rho.png')
	plt.savefig('figures/sanity/T-P-rho.pdf')

	plt.figure(figsize=(10,4))
	plt.plot(r/R_sun, nabla_stable, ls='dotted', color='black', label=r'$\nabla_{stable}$')
	plt.plot(r/R_sun, nabla_star, ls='solid', color='black', label=r'$\nabla^*$')
	plt.plot(r/R_sun, nabla_ad * np.ones(len(r)), ls=(0, (3, 5, 1, 5)), color='black', label=r'$\nabla_{ad}$')
	plt.ylabel(r'$\nabla$')
	plt.xlabel(r'$R/R_\odot$')
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig('figures/sanity/gradients.pdf')
	plt.savefig('figures/sanity/gradients.png')

	cross_section(r, L, F_con, sanity=True, savefig=True)

	fig1, axes = plt.subplots(2, 2, figsize=(16 * 2/3, 9 * 2/3))
	ax = axes.flatten()

	title_list = [r'$2R_0, 5R_0, 10R_0$', r'$2T_0, 5T_0, 10T_0$', \
				  r'$10^2\rho_0, 10^3\rho_0, 10^4\rho_0$', r'$10^2P_0, 10^3P_0, 10^4P_0$']

	label_list = [r'$M/M_{max}$', r'$R/R_{max}$', r'$L/L_{max}$']

	print_list = ['R', 'T', 'rho', 'P']

	for i in range(4):

		if i < 2: 

			factor_list = [2, 5, 10]
			factor_print = ['2', '5', '10']

		else:

			factor_list = [1e2, 1e3, 1e4]
			factor_print = ['1e2', '1e3', '1e4']

		for j in range(len(factor_list)):

			parameter_list = [R_0, T_0, rho_0, P_0]
			parameter_list[i] *= factor_list[j]
			R, T, rho, P = parameter_list

			test = Star(M_0, R, P, L_0, T)

			print(f'\nIntegrating for ' + factor_print[j] + print_list[i])

			test.integrate_equtations(p=1e-2)

			m, r, P, L, T, rho, nabla_star, nabla_stable, F_rad, F_con, eps = test.get_arrays()

			cross_section(r, L, F_con, title=factor_print[j] + print_list[i], savefig=True)

			iterations = np.linspace(0, len(m), len(m))		

			ax[i].plot(iterations, m/np.max(m), ls='solid', color='black')
			ax[i].plot(iterations, r/np.max(r), ls='dashed', color='black')
			ax[i].plot(iterations, L/np.max(L), ls='dashdot', color='black')

		ax[i].set_title(title_list[i])
		ax[i].legend(label_list)

	fig1.tight_layout()
	fig1.savefig('figures/testing/variable-params.pdf')
	fig1.savefig('figures/testing/variable-params.png')

	plt.show()