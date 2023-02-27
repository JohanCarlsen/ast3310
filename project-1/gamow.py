'''
In this program, we calculate the Gamow peak for 
all of the reactions in the PP branches and the 
CNO cycle, as functions of energy.
'''
import numpy as np 
import matplotlib.pyplot as plt
import scipy.constants as const

'''
Defining constants
'''
e = const.e 					# Elementary charge [C]
h = const.h 					# Planck constant [J Hz^-1]
pi = const.pi 					# Pi
k_B = const.k_B					# Boltzmann constant [JK^-1]
epsilon_0 = const.epsilon_0		# Vacuum permittivity


def lambda_sigma(m_i, m_k, Z_i, Z_k, energy):
	'''
	Function to compute the Gamow peak for the reaction
	between	element_i and element_k, at a given energy.
	Masses are m_i and m_k, atomic number is Z_i and Z_k.
	'''
	T = 1.57e7 		# Sun temperature [K]

	m = m_i * m_k / (m_i + m_k)		# Reduced mass 

	lmbda = np.exp(-energy / (k_B * T))
	sigma = np.exp(-np.sqrt(m / (2 * energy)) * Z_i * Z_k * e**2 * pi / (epsilon_0 * h))

	gamow = lmbda * sigma 
	idx = np.where([gamow == np.max(gamow)])[0][0]
	peak = gamow[idx]