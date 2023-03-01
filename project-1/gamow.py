'''
In this program, we calculate the Gamow peak for 
all of the reactions in the PP branches and the 
CNO cycle, as functions of energy.
'''
import numpy as np 
import matplotlib.pyplot as plt
import scipy.constants as const

plt.rcParams['lines.linewidth'] = 1 	# Setting linewidth for lines in plots

'''
Defining constants
'''
e = const.e 					# Elementary charge [C]
h = const.h 					# Planck constant [J Hz^-1]
pi = const.pi 					# Pi
eV = const.eV					# Electron volt [J]
m_u = const.m_u					# Atomic mass unit [kg]
k_B = const.k 					# Boltzmann constant [JK^-1]
epsilon_0 = const.epsilon_0		# Vacuum permittivity


def lambda_sigma(m_i, m_k, Z_i, Z_k, energy):
	'''
	Function to compute the Gamow peak for the reaction
	between	element_i and element_k, at a given energy.
	Masses are m_i and m_k, atomic number is Z_i and Z_k.
	'''
	T = 1.57e7 		# Sun temperature [K]

	m = m_i * m_k / (m_i + m_k)		# Reduced mass 

	'''
	The two exponentials from the proportionality function (lambda) and
	the cross section of the tunneling effect (sigma).
	'''
	lmbda = np.exp(-energy / (k_B * T))
	sigma = np.exp(-np.sqrt(m / (2 * energy)) * Z_i * Z_k * e**2 * pi / (epsilon_0 * h))

	'''
	The Gamow peak is found by the product of the proportionality
	function and the the tunneling effect cross section.
	'''
	gamow = lmbda * sigma
	idx = np.where([gamow == np.max(gamow)])[0][0]
	peak = gamow[idx]

	return gamow


'''
Defining element masses in atomic mass units
'''
H1 = 1.007825 * m_u			# Hydrogen mass in u 
D2 = 2.014 * m_u			# Deuterium mass in u 
He3 = 3.016 * m_u 			# Helium-3 mass in u 
He4 = 4.0026 * m_u			# Helium-4 mass in u
Be7 = 7.01693 * m_u			# Beryllium-7 mass in u
Li7 = 7.016004 * m_u		# Lithium-7 mass in u 

C12 = 12. * m_u				# Carbon-12 mass in u 
C13 = 13.00336 * m_u		# Carbon-13 mass in u 
N13 = 13.00574 * m_u		# Nitrogen-13 mass in u 
N14 = 14.00307 * m_u		# Nitrogen-14 mass in u 
N15 = 15.00109 * m_u		# Nitrogen-15 mass in u

'''
Defining atomic numbers
'''
Z_H = 1 		# Hydrogen
Z_He = 2 		# Helium
Z_Be = 4 		# Beryllium
Z_Li = 3 		# Lithium

Z_C = 6 		# Carbon
Z_N = 7 		# Nitrogen
Z_O = 8 		# Oxygen

'''
Defining energy interval [10^-17, 10^-13] J
'''
N = 1001
E = np.logspace(-17, -13, N)

rel_prob = np.zeros((10, N))	# Array to hold all of the probabilities 

PP0_1 = lambda_sigma(H1, H1, Z_H, Z_H, E)
PP0_2 = lambda_sigma(D2, H1, Z_H, Z_H, E)

PP1 = lambda_sigma(He3, He3, Z_He, Z_He, E)

PP2_1 = lambda_sigma(He3, He4, Z_He, Z_He, E)
PP2_2 = lambda_sigma(Li7, H1, Z_Li, Z_H, E)

PP3 = lambda_sigma(Be7, H1, Z_Be, Z_H, E)

CNO_1 = lambda_sigma(C12, H1, Z_C, Z_H, E)
CNO_2 = lambda_sigma(C13, H1, Z_C, Z_H, E)
CNO_3 = lambda_sigma(N14, H1, Z_N, Z_H, E)
CNO_4 = lambda_sigma(N15, H1, Z_N, Z_H, E)

tot_prob = np.sum(PP0_1 + PP0_2 + PP1 + PP2_1 + PP2_2 + PP3 + CNO_1 + CNO_2 + CNO_3 + CNO_4)

rel_prob[0, :] = PP0_1 / tot_prob 
rel_prob[1, :] = PP0_2 / tot_prob 
rel_prob[2, :] = PP1 / tot_prob
rel_prob[3, :] = PP2_1 / tot_prob
rel_prob[4, :] = PP2_2 / tot_prob
rel_prob[5, :] = PP3 / tot_prob
rel_prob[6, :] = CNO_1 / tot_prob
rel_prob[7, :] = CNO_2 / tot_prob
rel_prob[8, :] = CNO_3 / tot_prob
rel_prob[9, :] = CNO_4 / tot_prob

labels = ['pp', 'pd', '33', '34', '17´', '17', 'p12', 'p13', 'p14', 'p15']

plt.figure(figsize=(10, 5))

for i in range(10):

	plt.plot(E / eV, rel_prob[i, :], label=labels[i])

plt.xlabel('Energy [eV]')
plt.ylabel('Relative probability')
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e-25, 5e-2])

plt.legend()
plt.savefig('figures/gamow.pdf')
plt.savefig('figures/gamow.png')
plt.show()







