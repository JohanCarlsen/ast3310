import sys
sys.path.append("../stellar-engines")
from energy_production import EnergyProduction
from energy_transport import Star 
from cross_section import cross_section
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp
import scipy.constants as const 

plt.rcParams['lines.linewidth'] = 1

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
max_param_vals = [.05, .05, .05]
max_core_lum = .995 
min_core_dist = .1 
min_con_zone_width = .15

n_params = 4 
n_tests = 4

goal_param_vec = np.array(max_param_vals + [max_core_lum, min_core_dist, min_con_zone_width])
model_param_array = np.zeros((n_params, n_tests + 2))

result_array = np.zeros((n_params, n_tests))
init_params = np.array([M_0, R_0, L_0, T_0])

print('\nPrograssion:')

for i in range(n_params):

	print(f'Starting on parameter {i+1} of {n_params}\n')

	for j in range(n_tests):

		k = j + 1 

		variable_params = init_params
		variable_params[i] *= k

		m_i, r_i, L_i, T_i = variable_params

		P_rad = 4/3 * const.sigma * T_i**4 / const.c 
		P_G = rho_0 / (mu * const.m_u) * const.k * T_i
		P = P_G + P_rad

		model = Star(m_i, r_i, P, L_i, T_i)
		model.integrate_equtations(p=1e-2, output=False)

		m, r, P, L, T, rho, nabla_star, nabla_stable, F_rad, F_con, eps = model.get_arrays()


		model_param_vals = [L[-1]/L[0], m[-1]/m[0], r[-1]/r[0]]
		model_core_lum = L[-1] / L[0]
		core_dist = r[(L < L[0] * .995)] / r[0]

		if core_dist.size == 0:

			model_core_dist = 0

		else:

			model_core_dist = core_dist[0]

		model_con_zone_range = r[np.logical_and(F_con > 0, r > r[0] / 2)]
		model_con_zone_width = (model_con_zone_range[0] - model_con_zone_range[-1]) / r[0]
		model_params = np.array(model_param_vals + [model_core_lum, model_core_dist, model_con_zone_width])

		squares = (np.sum(goal_param_vec * model_params) - (np.sum(goal_param_vec) * np.sum(model_params))) \
														 / (np.sum(goal_param_vec**2) - np.sum(goal_param_vec)**2)

		result_array[i, j] = squares

		print(f'Finished {j+1} of {n_tests} tests on paramter {i+1}')

	print(f'\nFinished testing for parameter {i+1} of {n_params}\n')


m_coeff = 1.45
_, r_coeff, L_coeff, T_coeff = np.min(result_array[:, 1:], axis=1)

M_fit = M_0 * m_coeff
R_fit = R_0 * r_coeff
L_fit = L_0 * L_coeff
T_fit = T_0 * T_coeff

best = Star(M_fit, R_fit, P_0, L_fit, T_fit)
best.integrate_equtations(p=1e-2)

m, r, P, L, T, rho, nabla_star, nabla_stable, F_rad, F_con, eps = best.get_arrays()

L_core = L[-1]/L[0]
R_core = r[(L < .995 * L[0])][0] / r[0]
con_width_range = r[np.logical_and(F_con > 0, r > r[0]/2)]
con_width = (con_width_range[0] - con_width_range[-1]) / r[0]

print(f'\nCore luminosity:\t{L_core:5.3f} * L_0')
print(f'Core radius:\t\t{R_core:7.3f} % of R_0')
print(f'Surface con. width:\t{con_width:7.3f} % of R_0\n')

iterations = np.linspace(0, len(m), len(m))

plt.figure(figsize=(10,4))
plt.plot(iterations, m/np.max(m), ls='solid', color='black', label=r'$M/M_{max}$')
plt.plot(iterations, r/np.max(r), ls='dashed', color='black', label=r'$R/R_{max}$')
plt.plot(iterations, L/np.max(L), ls='dashdot', color='black', label=r'$L/L_{max}$')
plt.plot(iterations, 0.05*np.ones(len(iterations)), ls='dotted', color='black')
plt.legend()
plt.xlabel('Iterations')
plt.tight_layout()
# plt.savefig('figures/sanity/convergence_check.pdf')
# plt.savefig('figures/sanity/convergence_check.png')

plt.figure()
plt.plot(r/np.max(r), eps, lw=1, color='black', label=r'$\partial L/\partial m$')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$R/R_{max}$')
plt.ylabel(r'Js$^{-1}$')
plt.tight_layout()
# plt.savefig('figures/sanity/epsilon.pdf')
# plt.savefig('figures/sanity/epsilon.png')

plt.figure(figsize=(10,4))
plt.plot(r/np.max(r), T/np.max(T), ls='dashed', color='black', label=r'$T/T_{max}$')
plt.plot(r/np.max(r), P/np.max(P), ls='dotted', color='black', label=r'$P/P_{max}$')
plt.plot(r/np.max(r), rho/np.max(rho), ls=(0, (3, 5, 1, 5)), color='black', label=r'$\rho/\rho_{max}$')
plt.legend()
plt.xlabel(r'$R/R_{max}$')
plt.tight_layout()
# plt.savefig('figures/sanity/T-P-rho.png')
# plt.savefig('figures/sanity/T-P-rho.pdf')

plt.figure(figsize=(10,4))
plt.plot(r/R_sun, nabla_stable, ls='dotted', color='black', label=r'$\nabla_{stable}$')
plt.plot(r/R_sun, nabla_star, ls='solid', color='black', label=r'$\nabla^*$')
plt.plot(r/R_sun, nabla_ad * np.ones(len(r)), ls=(0, (3, 5, 1, 5)), color='black', label=r'$\nabla_{ad}$')
plt.ylabel(r'$\nabla$')
plt.xlabel(r'$R/R_\odot$')
plt.yscale('log')
plt.legend()
plt.tight_layout()
# plt.savefig('figures/sanity/gradients.pdf')
# plt.savefig('figures/sanity/gradients.png')

# cross_section(r, L, F_con, sanity=True, savefig=True)
cross_section(r, L, F_con)

plt.show()






