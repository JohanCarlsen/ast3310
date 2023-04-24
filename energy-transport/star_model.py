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
from scipy.stats import chisquare as chi

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
rho_0 = 1.42e-7 * avgRho_sun * 30
T_0 = 5770
P_G = rho_0 / (mu * const.m_u) * const.k * T_0		# Gas pressure 
P_rad = 4/3 * const.sigma * T_0**4 / const.c 		# Radiative pressure 
P_0 = (P_G + P_rad)
L_0 = L_sun
M_0 = M_sun
R_0 = R_sun

variable_params = np.array([M_0, R_0, L_0, T_0])

M_fit, R_fit, L_fit, T_fit = variable_params * [1, 0.9, 1.2, .8]

best = Star(M_fit, R_fit, P_0, L_fit, T_fit)
best.integrate_equtations(p=1e-2, include_cycles=True)

m, r, P, L, T, rho, nabla_star, nabla_stable, F_rad, F_con, eps, PP1, PP2, PP3, CNO = best.get_arrays(include_cycles=True)

r_plot = r/R_sun
ticks = np.linspace(r_plot[-1], r_plot[0], 6)

F_con_fraction = F_con / np.sum(F_rad + F_con)
F_rad_fraction = F_rad / np.sum(F_rad + F_con)

eps_max = PP1 + PP2 + PP3 + CNO
# eps_max += (1 * (eps_max == 0))

L_core = L[-1]/L[0]
R_core = r[(L < .995 * L[0])][0] / r[0]
con_width_range = r[np.logical_and(F_con > 0, r > r[0]/2)]
con_width = (con_width_range[0] - con_width_range[-1]) / r[0]

print(f'\nCore luminosity:\t{L_core*100:7.2e} % of L_0')
print(f'Core radius:\t\t{R_core*100:7.2e} % of R_0')
print(f'Surface con. width:\t{con_width*100:7.2e} % of R_0\n')

gridspec = dict(hspace=0, height_ratios=[1, 1, .4, 1])
fig, axes = plt.subplots(4, 1, figsize=(9, 7), gridspec_kw=gridspec)

axes[0].plot(r_plot, T, ls='solid', color='black', label=r'$T$')
axes[0].legend()
axes[0].set_xticklabels([])

axes[1].plot(r_plot, m/M_sun, ls='solid', color='black', label=r'$M/M_\odot$')
axes[1].plot(r_plot, L/L_sun, ls='dashed', color='black', label=r'$L/L_\odot$')
axes[1].legend()
axes[1].set_xticks(ticks)
axes[1].set_xlabel(r'$R/R_\odot$')

axes[2].set_visible(False)

axes[3].plot(r_plot, rho/avgRho_sun, ls='solid', color='black', label=r'$\rho/\bar{\rho}_\odot$')
axes[3].plot(r_plot, P, ls='dashed', color='black', label=r'$P$')
axes[3].legend()
axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].set_xlabel(r'$\log_{10}(R/R_\odot)$')
fig.savefig('figures/best_fit_T-M-L-rho-P.pdf')
fig.savefig('figures/best_fit_T-M-L-rho-P.png')

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(r_plot, F_con_fraction, ls='solid', color='black', label=r'$F_{con}$')
ax.plot(r_plot, F_rad_fraction, ls='dashed', color='black', label=r'$F_{rad}$')
ax.legend()
ax.set_xlabel(r'$R/R_\odot$')
ax.set_ylabel(r'Fraction of $F_i$')
ax.set_xticks(ticks)
fig.tight_layout()
fig.savefig('figures/flux-fraction.pdf')
fig.savefig('figures/flux-fraction.png')

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(r_plot[(eps_max != 0)], np.flip(PP1/eps_max)[(eps_max != 0)], ls='dashed', color='black', label=r'$\epsilon_{PP1}/\epsilon_{max}$')
ax.plot(r_plot[(eps_max != 0)], np.flip(PP2/eps_max)[(eps_max != 0)], ls=(0, (1, 10)), color='black', label=r'$\epsilon_{PP2}/\epsilon_{max}$')
ax.plot(r_plot[(eps_max != 0)], np.flip(PP3/eps_max)[(eps_max != 0)], ls=(0, (5, 10)), color='black', label=r'$\epsilon_{PP3}/\epsilon_{max}$')
ax.plot(r_plot[(eps_max != 0)], np.flip(CNO/eps_max)[(eps_max != 0)], ls='dotted', color='black', label=r'$\epsilon_{CNO}/\epsilon_{max}$')
ax.plot(r_plot[(eps_max != 0)], (eps/np.max(eps))[(eps_max != 0)], ls='solid', color='black', label=r'$L/\epsilon_{max}$')
ax.legend()
ax.set_xticks(ticks)
ax.set_xlabel(r'$R/R_\odot$')
ax.set_ylabel(r'Rel. energy')
fig.tight_layout()
fig.savefig('figures/rel-energy.pdf')
fig.savefig('figures/rel-energy.png')

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(r_plot, nabla_star, ls='solid', color='black', label=r'$\nabla^*$')
ax.plot(r_plot, nabla_stable, ls='dashed', color='black', label=r'$\nabla_{stable}$')
ax.plot(r_plot, nabla_ad * np.ones(len(r_plot)), ls='dotted', color='black', label=r'$\nabla_{ad}$')
ax.legend()
ax.set_xlabel(r'$R/R_\odot$')
ax.set_ylabel(r'$\nabla_i$')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('figures/gradients-final.pdf')
fig.savefig('figures/gradients-final.png')

cross_section(r, L, F_con, savefig=True)

plt.show()