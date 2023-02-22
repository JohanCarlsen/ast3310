'''
In this program we calculate the energy production in a star, given its 
temperature and density.
'''
import numpy as np 
import scipy.constants as const 

class EnergyProduction:
	'''
	Class instance takes temperature in K and density in kgm^-3 as 
	input arguments.
	'''
	def __init__(self, temperature, density):

		self.T = temperature
		self.rho = density

		'''
		Arrays will contain the energy production rate for each of 
		the reactions in all of the PP branches and the CNO cycle.
		'''
		self.PP0 = np.zeros(1)
		self.PP1 = np.zeros(1)
		self.PP2 = np.zeros(3)
		self.PP3 = np.zeros(4)
		self.CNO = np.zeros(6)

		'''
		Mass fractions of the atomic species.
		'''
		self.X = .7 		# Hydrogen
		self.Y = .29 		# Helium-4
		self.Z73Li = 1e-7 	# Litium-7
		self.Z74Be = 1e-7 	# Berrylium-7
		self.Y32He = 1e-10 	# Helium-3
		self.Z147N = 1e-11 	# Nitrogen-14

		'''
		Energy release accociated with the different fusion
		reactions in J.
		'''
		self.Q = {
		'pp': 1.177e6 * const.eV,
		'pd': 5.494e6 * const.eV,
		'33': 12.860e6 * const.eV,
		'34': 1.586e6 * const.eV,
		'e7': .049e6 * const.eV,
		'17_mark': 17.346e6 * const.eV,
		'17': .137e6 * const.eV,
		'8': 8.367e6 * const.eV,
		'8_mark': 2.995e6 * const.eV
		}


	def reaction_rates(self, temperature):
		'''
		Function will take temperature in K as argument and return a dictionary
		containing the energy rates (lambdas) for each reaction.
		'''
		T9 = temperature * 1e-9			# Temperature in giga K
		T9x = T9 / (1 + 4.95e-2 * T9)
		T9xx = T9 / (1 + .759 * T9)

		N_A = const.N_A * 1e6			# Avogrado's number was given in cm^3 mol^-1, adjusting to m^3 mol^-1

		lmbda = {
		'pp': (4.01e-15 * T9**(-2/3) * np.exp(-3.38 * T9**(-1/3)) * (1 + .123 * T9**(1/3) + 1.09 * T9**(2/3) + .938 * T9)) / N_A,
		'33': (6.04e10 * T9**(-2/3) * np.exp(-12.276 * T9**(-1/3)) * (1 + .034 * T9**(1/3) - .522 * T9**(2/3) - .124 * T9 + .353 * T9**(4/3) + .213 * T9**(5/3))) / N_A,
		'34': (5.61e6 * T9x**(5/6) * T9**(-2/3) * np.exp(-12.826 * T9x**(-1/3))) / N_A,
		'e7': (1.34e-10 * T9**(-1/2) * (1 - .537 * T9**(1/3) + 3.86 * T9**(2/3) + .0027 * T9**(-1) * np.exp(2.515e-3 * T9**(-1)))) / N_A,
		'17_mark': (1.096e9 * T9**(-2/3) * np.exp(-8.472 * T9**(-1/3)) - 4.83e8 * T9xx**(5/6) * T9**(-3/2) * np.exp(-8.472 * T9xx**(-1/3)) + 1.06e10 * T9**(-3/2) * np.exp(-30.442 * T9**(-1))) / N_A,
		'17': (3.11e5 * T9**(-2/3) * np.exp(-10.262 * T9**(-1/3)) + 2.53e3 * T9**(-2/3) * np.exp(-7.306 * T9**(-1))) / N_A,
		'p14': (4.9e7 * T9**(-2/3) * np.exp(-15.228 * T9**(-1/3) - .092 * T9**2) * (1 + .027 * T9**(1/3) - .778 * T9**(2/3) - .149 * T9 + .261 * T9**(4/3) + .127 * T9**(5/3)) + 2.37e3 * T9**(-3/2) * np.exp(-3.011 * T9**(-1)) + 2.19e4 * np.exp(-12.53 * T9**(-1))) / N_A
		}

		return lmbda


	def number_density(self, density):
		'''
		Function will take density in kgm^-3 as argument and return dictionary
		containing the number densities of each element.
		'''
		rho = density

		n = {
		'n_p': rho * self.X / const.m_u,
		'n_He': rho * self.Y / (4 * const.m_u),
		'n_32He': rho * self.Y32He / (3 * const.m_u),
		'n_73Li': rho * self.Z73Li / (7 * const.m_u),
		'n_74Be': rho * self.Z74Be / (7 * const.m_u),
		'n_e': rho / (2 * const.m_u) * (1 + self.X),
		}

		return n 


	def r(self, n_i, n_k, lmbda, rho):
		'''
		Method to calculate the reaction rate of element i and k, given
		their number density and accociated lambda value.
		'''
		if n_i == n_k:

			delta = 1.

		else:

			delta = 0

		rate = n_i * n_k * lmbda / (rho * (1 + delta))

		return rate 


	def pp0(self, PP0_array=None, temperature=None, density=None):
		'''
		Method to compute the energy production rate of the PP0 cycle
		'''
		if PP0_array is None:

			PP0_array = self.PP0

		if temperature is None:

			T = self.T 

		if density is None:

			rho = self.rho

		n_i = self.number_density(rho)['n_p']	# Number density of protons
		n_k = n_i
		Q_i , Q_k = self.Q['pp'], self.Q['pd']	# Energy released by fusion of protons to deuterium and deuterium and proton to helium-3
		lmbda = self.reaction_rates(T)['pp']	# Reaction rate for two protons

		r_ik = self.r(n_i, n_k, lmbda, rho)
		eps = r_ik * (Q_i + Q_k)

		PP0_array[0] = eps


	def pp1(self, PP1_array=None, temperature=None, density=None):
		'''
		Method to compute the energy production rate of the PP1 cycle
		'''
		if PP1_array is None:

			PP1_array = self.PP1

		if temperature is None:

			T = self.T 

		if density is None:

			rho = self.rho

		n_i = self.number_density(rho)['n_32He']	# Number density of helium-3
		n_k = n_i 
		Q_ik = self.Q['33']							# Energy released by fusing two helium-3 to helium-4
		lmbda = self.reaction_rates(T)['33']		# Reaction rate for two helium-3

		r_ik = self.r(n_i, n_k, lmbda, rho)
		eps = r_ik * Q_ik

		PP1_array[0] = eps


	def pp2(self, PP2_array=None, temperature=None, density=None):
		'''
		Method to compute the energy production rate of the PP2 cycle
		'''
		if PP2_array is None:

			PP2_array = self.PP2 

		if temperature is None:

			T = self.T 

		if density is None:

			rho = self.rho 

		n = self.number_density(rho)
		n_i = np.array([n['n_32He'], n['n_74Be'], n['n_73Li']])				# Number densities of helium-3, beryllium-7, and litium-7
		n_k = np.array([n['n_He'], n['n_e'], n['n_p']])						# Number densities of helium-4, electrones, and protons
		Q_ik = np.array([self.Q['34'], self.Q['e7'], self.Q['17_mark']])	# Energy releases of the fusion reactions 
		lmbda = self.reaction_rates(T)
		lmbda_ik = np.array([lmbda['34'], lmbda['e7'], lmbda['17_mark']])	# Reaction rates for the PP2 branch

		for i in range(3):

			r_ik = self.r(n_i[i], n_k[i], lmbda_ik[i], rho)
			eps = r_ik * Q_ik[i]

			PP2_array[i] = eps







T_sun = 1.57e7
rho_sun = 1.62e5
a = EnergyProduction(T_sun, rho_sun)
a.pp0()
a.pp1()
a.pp2()
print('Sanity check:')
print(f'PP0:\t{a.PP0[0]*a.rho:.3e}')
print(f'PP1:\t{a.PP1[0]*a.rho:.3e}')
print(f'PP2:\t{a.PP2[0]*a.rho:.3e}')
print(f'\t{a.PP2[1]*a.rho:.3e}')
print(f'\t{a.PP2[2]*a.rho:.3e}')



