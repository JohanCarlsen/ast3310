'''
This program will calculate the mass difference energy output
from the PP branches and the CNO cycle, and compare them with 
the known output. Then we can find the neutrino energy as the 
difference between the two.
'''
import numpy as np 
import scipy.constants as const 

def energy_output(reaction):
		'''
		Function to calculate tne energy output from dE=dm c^2.
		The array_type reaction is on the form (initial_mass, final_mass),
		and is assumed to be in units of u.
		'''
		MeV_c2 = 931.4941			# Atomic mass unit in MeV c^-2
		init_mass = reaction[0]
		final_mass = reaction[1]

		dE = -(final_mass - init_mass) * MeV_c2

		return dE 


'''
Defining constants and elements in atomic mass units
'''
u = const.m_u			# Atomic mass unit in kg 
m_e = const.m_e / u 	# Electron mass in u
m_p = const.m_p / u 	# Proton mass in u 

H1 = 1.007825			# Hydrogen mass in u 
D2 = 2.014				# Deuterium mass in u 
He3 = 3.016 			# Helium-3 mass in u 
He4 = 4.0026			# Helium-4 mass in u
Be7 = 7.01693			# Beryllium-7 mass in u
Be8 = 8.00531			# Beryllium-8 mass in u
Li7 = 7.016004			# Lithium-7 mass in u 
B8 = 8.02461			# Boron-8 mass in u 

C12 = 12.				# Carbon-12 mass in u 
C13 = 13.00336			# Carbon-13 mass in u 
N13 = 13.00574			# Nitrogen-13 mass in u 
N14 = 14.00307			# Nitrogen-14 mass in u 
N15 = 15.00109			# Nitrogen-15 mass in u
O15 = 15.00307			# Oxygen-15 mass in u 

'''
Defining arrays that will contain the 
PP cranches and the CNO cycle.
'''
PP_branches = np.zeros((10, 2))
PP_branches[0, :] = np.array([2 * H1, D2])				# H1+H1 -> D2+e(+)		(Removed positron mass, added annihilation energy)
PP_branches[1, :] = np.array([D2 + H1, He3])			# D2+H1 -> He3
PP_branches[2, :] = np.array([2 * He3, He4 + 2 * H1])	# He3+He3 -> He4+2H1
PP_branches[3, :] = np.array([He3 + He4, Be7])			# He3+He4 -> Be7
PP_branches[4, :] = np.array([Be7, Li7])				# Be7+e -> Li7
PP_branches[5, :] = np.array([Li7 + H1, 2 * He4])		# Li7+H1 -> 2He4
PP_branches[6, :] = np.array([He3 + He4, Be7])			# He3+He4 -> Be7
PP_branches[7, :] = np.array([Be7 + H1, B8])			# Be7+H1 -> B8
PP_branches[8, :] = np.array([B8, Be8])					# B8 -> Be8+e(+) 		(Removed positron mass, added annihilation energy)
PP_branches[9, :] = np.array([Be8, 2 * He4])			# Be8 -> 2He4

PP_energy_output = np.zeros(10)

'''
Calculating all enrgy outputs
from all of the PP branches.
'''
for i in range(10):

	reaction = PP_branches[i, :]
	e_out = energy_output(reaction)

	PP_energy_output[i] = e_out

'''
This array will hold all of the energies.
'''
PP_Q_neutrino = np.zeros((10, 2))
PP_Q_neutrino[0, :] = np.array([1.177, .265])		# Neutrino created
PP_Q_neutrino[1, :] = np.array([5.494, 0])			
PP_Q_neutrino[2, :] = np.array([12.86, 0])			
PP_Q_neutrino[3, :] = np.array([1.586, 0])			
PP_Q_neutrino[4, :] = np.array([.049, .815])		# Neutrino created 
PP_Q_neutrino[5, :] = np.array([17.346, 0])			
PP_Q_neutrino[6, :] = np.array([1.586, 0])			
PP_Q_neutrino[7, :] = np.array([.137, 0])			
PP_Q_neutrino[8, :] = np.array([8.367, 6.711])		# Neutrino created 
PP_Q_neutrino[9, :] = np.array([2.995, 0])			


'''
CNO cycle
'''
CNO_cycle = np.zeros((6, 2))
CNO_cycle[0, :] = np.array([C12 + H1, N13])			# C12+H1 -> N13
CNO_cycle[1, :] = np.array([N13, C13])				# N13 -> C13+e(+)		(Removed positron mass, added annihilation energy)
CNO_cycle[2, :] = np.array([C13 + H1, N14])			# C13+H1 -> N14
CNO_cycle[3, :] = np.array([N14 + H1, O15])			# N14+H1 -> O15
CNO_cycle[4, :] = np.array([O15, N15])				# O15 -> N15+e(+)		(Removed positron mass, added annihilation energy)
CNO_cycle[5, :] = np.array([N15 + H1, C12 + He4])	# N15+H1 -> C12+He4

CNO_energy_output = np.zeros(6)

'''
Calculating the energy output
from the CNO cycle.
'''
for i in range(6):

	reaction = CNO_cycle[i, :]
	e_out = energy_output(reaction)

	CNO_energy_output[i] = e_out

CNO_Q_neutrino = np.zeros((6, 2))
CNO_Q_neutrino[0, :] = np.array([1.944, 0])
CNO_Q_neutrino[1, :] = np.array([1.513, .707])		# Neutrino created 
CNO_Q_neutrino[2, :] = np.array([7.551, 0])
CNO_Q_neutrino[3, :] = np.array([7.297, 0])
CNO_Q_neutrino[4, :] = np.array([1.757, .997])		# Neutrino created
CNO_Q_neutrino[5, :] = np.array([4.966, 0])

'''
Since we have the actual energy outputs (Q and neutrino), we can 
print the results and compare to exact energies.
'''
PP0 = np.sum(PP_energy_output[:2])		# PP0 is common for all the PP branches
Q0 = np.sum(PP_Q_neutrino[:2, 0])

branch_totals = {
'PP1': [f'{2 * (PP0) + PP_energy_output[2]:.3f}', f'{2 * (Q0) + PP_Q_neutrino[2, 0]:.3f}'],				# Adding 2*the PP0 for PP1
'PP2': [f'{PP0 + np.sum(PP_energy_output[3:6]):.3f}', f'{Q0 + np.sum(PP_Q_neutrino[3:6, 0]):.3f}'],		# Adding PP0 to both PP2 and PP3
'PP3': [f'{PP0 + np.sum(PP_energy_output[6:]):.3f}', f'{Q0 + np.sum(PP_Q_neutrino[6:, 0]):.3f}'],
'CNO': [f'{np.sum(CNO_energy_output):.3f}', f'{np.sum(CNO_Q_neutrino[:, 0]):.3f}']
}

print(f"\n{'Branch':<5} | {'Mass diff. [MeV]':<14} | {'Ex. [MeV]':<10} | {'Neutrino [MeV]':<14} | {'Lost to nu'}")

for key, vals in branch_totals.items():

	m_diff, m_ex = vals									# Energy release due to mass difference
	neut = f'{float(m_diff) - float(m_ex):.3f}'			# Neutrino energy is the diff. betw. exact and mass difference
	perc = f'{float(neut) / float(m_diff) * 100:.2f}'	# Percentage lost due to neutrinos

	print(f'{key:<6} | {m_diff:<16} | {m_ex:<10} | {neut:<14} | {perc:<5} %')
