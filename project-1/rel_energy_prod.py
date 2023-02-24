'''
In this program we make instances of the class EnergyProduction,
and calculate and plot the relative energy production from the
PP branches and the CNO cycle.
'''
import numpy as np 
import matplotlib.pyplot as plt 
from energy_production import EnergyProduction

T_range = np.linspace(1e4, 1e9, 101)	# Temperatures in K
rho_sun = 1.62e5						# Sun density in kgm^-3

energies = np.zeros((4, len(T_range)))

for i in range(len(T_range)):

	T = T_range[i]
	star = EnergyProduction(T, rho_sun)
	star.run_all_cycles()
