import sys
sys.path.append("../stellar-engines")
from energy_production import EnergyProduction
from cross_section import cross_section
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp
import scipy.constants as const 

plt.rcParams['lines.linewidth'] = 1

# Extracting data from opacity.txt.
log_R = np.loadtxt('opacity.txt', usecols=range(1, 20), max_rows=1)		# [gcm^-3 K^-3]
data = np.loadtxt('opacity.txt', skiprows=2)
log_T = data[:, 0]														# [K]
log_K = data[:, 1:]														# [cm^2 g^-1]

spline = RectBivariateSpline(log_T, log_R, log_K)	# Spline containing the interpolated values of the opacity

def opacity_sanity_check():
	'''
	Perform a sanity check, and print the result to terminal.
	The values for T and R are given in the project description.
	'''
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


def opacity(rho, T, get_warnings=False):
	'''
	Taked density and temperature in SI-units, and computes the	opacity with either
	interpolating or extrapolating from the	spline object. A warning will be printed
	if either of the values	for density or temperature is outside of the interval of
	the spline,	ie. extrapolating.
	'''
	rho *= 1e-3
	log_R = np.log10(rho / (T / 10**6)**3)
	log_T = np.log10(T)

	log_K = spline.ev(log_T, log_R)
	K = 10**log_K * 1e-1

	if get_warnings:

		if log_T < 3.75 or log_T > 8.7:

			print('\nWarning: Entered value for log(T) resulting in extrapolation.')
			print(f'Entered value for log(T): {log_T:g}')

		if log_R < -8 or log_R > 1:

			print('\nWarning: Entered value for log(R) resulting in extrapolation.')
			print(f'Entered value for log(R): {log_R:g}\n')	

	return K


def get_density_pressure(T=5770., density=None, pressure=None, get_info=False):
	'''
	Method to calculate the density rho(P,T) or pressure P(rho,T), given 
	the temperature T in Kelvin (default at 5770 K), and either rho or P.
	'''
	if density is None and pressure is None:

		print(f"{'Error:':<7}Missing parameter input!")
		print(f"{'':<7}Either density or pressure needs to be provided!")
		exit()

	P_rad = 4 / (3 * const.c) * const.sigma * T**4

	if density is not None:

		P_gas = density / (mu * const.m_u) * const.k * T
		P = P_rad + P_gas 

		if get_info:

			print(f"\nComputed pressure from get_density_pressure: {P:.2e} Pa\n")

		return P 

	if pressure is not None: 

		rho = (pressure - P_rad) * mu * const.m_u / (const.k * T)

		if get_info:

			print(f"\nComputed density from get_density_pressure: {rho:.2e} kg/m\u00b3\n")

		return rho 
	

def integrate(variables, p=1e-2, sanity_check=False):
	'''
	This function takes an array of variables and integrates them one 
	step, with variable step length. The default argument p ensures 
	that none of the variables (V) change by more then pV.
	'''
	if sanity_check:

		rho = 55.9
		kappa = 3.98

		m, r, T = variables

	else:

		m, r, P, L, T = variables	

		rho = get_density_pressure(T, pressure=P)
		kappa = opacity(rho, T)

	c_P = 5/2 * const.k / (mu * const.m_u)
	g = const.G * m / r**2
	H_P = const.k * T / (mu * const.m_u * g)
	l_m = H_P
	r_p = l_m / 2
	SQd = 2 / r_p

	U = 64 * const.sigma * T**3 / (3 * kappa * rho**2 * c_P) * np.sqrt(H_P / g)

	K = U * SQd / l_m

	if sanity_check:

		nabla_stable = 3.26078

	else:

		nabla_stable = 3 * kappa * rho * H_P * L / (64 * const.sigma * T**4 * const.pi * r**2)

	coeffs = [l_m**2/U, 1, K, (nabla_ad - nabla_stable)]
	roots = np.roots(coeffs)
	real_root_idx = np.where(roots.imag == 0)[0][0]
	xi = roots[real_root_idx].real

	nabla_star = xi**2 + xi * K + nabla_ad

	F_rad = 16 * const.sigma * T**4 / (3 * kappa * rho * H_P) * nabla_star
	F_con = rho * c_P * T * np.sqrt(g) * H_P**(-3/2) * (l_m/2)**2 * xi**3

	star = EnergyProduction(T, rho)
	star.run_all_cycles()
	eps = star.get_total_energy() #/ 4.3e2

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

		opacity_sanity_check()

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

	if nabla_stable > nabla_ad:

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
	M_new = m - dm

	return M_new, r_new, P_new, L_new, T_new, rho, nabla_stable, nabla_star, F_rad, F_con, eps

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

nabla_ad = 2/5 			# Adiabatic temperature gradient for ideal gas

sanity_T = 9e5 
sanity_R = 0.84 * R_sun
sanity_M = 0.99 * M_sun

sanity_variables = np.array([sanity_M, sanity_R, sanity_T])

# Initial parameters 
rho_0 = 1.42e-7 * avgRho_sun
T_0 = 5770
P_G = rho_0 / (mu * const.m_u) * const.k * T_0		# Gas pressure 
P_rad = 4/3 * const.sigma * T_0**4 / const.c 		# Radiative pressure 
P_0 = (P_G + P_rad)
L_0 = L_sun
M_0 = M_sun
R_0 = R_sun
	
# Lists to hold the evolved values 
mass = [M_0]
radius = [R_0]
pressure = [P_0]
luminosity = [L_0]
temperature = [T_0]
density = [rho_0]

# Gradient lists
nabla_stable = []
nabla_star = []

# Flux lists
rad_flux = []
con_flux = []

epsilon = []

i = 0 
while mass[i] > 0 and radius[i] > 0 \
				  and luminosity[i] > 0 \
				  and i < 4.5e3:
	'''
	The while-loop runs until the radius
	is no longer larger than zero. 
	'''
	variables = np.array([mass[i], radius[i], pressure[i], luminosity[i], temperature[i]])
	M_new, r_new, P_new, L_new, T_new, rho, nabla_stable_, nabla_star_, F_rad, F_con, eps = integrate(variables)
	
	mass.append(M_new)
	radius.append(r_new)
	pressure.append(P_new)
	luminosity.append(L_new)
	temperature.append(T_new)
	density.append(rho)

	nabla_stable.append(nabla_stable_)
	nabla_star.append(nabla_star_)
	rad_flux.append(F_rad)
	con_flux.append(F_con)

	epsilon.append(eps)

	i += 1

# Converting the lists to numpy-arrays
M = np.array(mass[:-1])						# Mass
R = np.array(radius[:-1])					# Radius
P = np.array(pressure[:-1])					# Pressure 
L = np.array(luminosity[:-1])				# Luminosity
T = np.array(temperature[:-1])				# Temperature 
rho = np.array(density[:-1])				# Density
grad_star = np.array(nabla_star)			# Star temperature gradient 
grad_stable = np.array(nabla_stable)		# Stable temperature gradient
F_rad = np.array(rad_flux)					# Radiative flux
F_con = np.array(con_flux)					# Convective flux 

iterations = np.linspace(0, len(M), len(M))

plt.figure(figsize=(10,4))
plt.plot(iterations, M/np.max(M), ls='solid', color='black', label=r'$M/M_{max}$')
plt.plot(iterations, R/np.max(R), ls='dashed', color='black', label=r'$R/R_{max}$')
plt.plot(iterations, L/np.max(L), ls='dashdot', color='black', label=r'$L/L_{max}$')
plt.plot(iterations, 0.05*np.ones(len(iterations)), ls='dotted', color='black')
plt.legend()
plt.xlabel('Iterations')
plt.tight_layout()
# plt.savefig('figures/convergence_check.pdf')
# plt.savefig('figures/convergence_check.png')

plt.figure()
plt.plot(R/np.max(R), epsilon)
plt.yscale('log')

plt.figure(figsize=(10,4))
plt.plot(R/np.max(R), T/np.max(T), ls='dashed', color='black', label=r'$T/T_{max}$')
plt.plot(R/np.max(R), P/np.max(P), ls='dotted', color='black', label=r'$P/P_{max}$')
plt.plot(R/np.max(R), rho/np.max(rho), ls=(0, (3, 5, 1, 5)), color='black', label=r'$\rho/\rho_{max}$')
plt.legend()
plt.xlabel(r'$R/R_{max}$')
plt.tight_layout()
plt.savefig('figures/T-P-rho.png')
plt.savefig('figures/T-P-rho.pdf')

print(f'Final values after {i} iterations:')
print(f'M[-1]/M_0: {M[-1]/M_0*100:4.1f} %')
print(f'R[-1]/R_0: {R[-1]/R_0*100:4.1f} %')
print(f'L[-1]/L_0: {L[-1]/L_0*100:4.1f} %')

plt.figure(figsize=(10,4))
plt.plot(R/R_sun, grad_stable, ls='dotted', color='black', label=r'$\nabla_{stable}$')
plt.plot(R/R_sun, grad_star, ls='solid', color='black', label=r'$\nabla^*$')
plt.plot(R/R_sun, nabla_ad * np.ones(len(R)), ls=(0, (3, 5, 1, 5)), color='black', label=r'$\nabla_{ad}$')
plt.ylabel(r'$\nabla$')
plt.xlabel(r'$R/R_\odot$')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('figures/gradients.pdf')
plt.savefig('figures/gradients.png')

plt.show()

cross_section(R, L, F_con, savefig=True)