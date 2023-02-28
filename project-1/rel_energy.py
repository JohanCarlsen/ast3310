'''
In this program we will calculate and plot the relative energy
production from each of the PP branches and the CNO cycle.
'''
import numpy as np 
import matplotlib.pyplot as plt 
from energy_production import EnergyProduction

'''
For the relative energy we will use temperature from 10^4 to 10^9 K.
'''
rho_sun = 1.62e5	# Sun density [kgm^-3]
N_steps = 1001	 	# No. of points to solve for 

T = np.logspace(4, 9, N_steps)			# Temperature range [K]
E_rel = np.zeros((4, N_steps))			# Array to contain the rel. energies

for i in range(N_steps):
	'''
	Creating an instance of the class EnergyProduction for each 
	of the temperatures. Computing the total energy production,
	and storing the rel. energies to E_rel.
	'''
	star = EnergyProduction(T[i], rho_sun)
	star.run_all_cycles()

	E_tot = np.sum(star.PP1) + np.sum(star.PP2) + np.sum(star.PP3) + star.CNO

	E_rel[0, i] = np.sum(star.PP1) / E_tot		# PP1 rel. energy
	E_rel[1, i] = np.sum(star.PP2) / E_tot		# PP2 rel. energy
	E_rel[2, i] = np.sum(star.PP3) / E_tot		# PP3 rel. energy
	E_rel[3, i] = star.CNO / E_tot				# CNO rel. energy

plt.figure(figsize=(10, 4))

plt.plot(T, E_rel[0, :], lw=1, ls='solid', color='black', label='PP1')
plt.plot(T, E_rel[1, :], lw=1, ls='dotted', color='black', label='PP2')
plt.plot(T, E_rel[2, :], lw=1, ls='dashed', color='black', label='PP3')
plt.plot(T, E_rel[3, :], lw=1, ls='dashdot', color='black', label='CNO')

plt.xscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Rel. energy production')
plt.legend()

# plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('figures/rel-energy-prod.pdf')
plt.savefig('figures/rel-energy-prod.png')

plt.show()