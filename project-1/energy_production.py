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
		self.PP3 = np.zeros(2)
		self.CNO = np.zeros(1)

		'''
		Arrays that will contain the reaction rates for each of the 
		PP branches and the CNO cycle
		'''
		self.r_PP0 = np.zeros(1)
		self.r_PP1 = np.zeros(1)
		self.r_PP2 = np.zeros(3)
		self.r_PP3 = np.zeros(2)
		self.r_CNO = np.zeros(1)

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
		'8_mark': 2.995e6 * const.eV,
		'CNO': (1.944 + 1.513 + 7.551 + 7.297 + 1.757 + 4.966) * 1e6 * const.eV
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
		'34': (5.61e6 * T9x**(5/6) * T9**(-3/2) * np.exp(-12.826 * T9x**(-1/3))) / N_A,
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
		'n_e': rho / const.m_u * (self.X + self.Y / 2 + 2/3 * self.Y32He + 3/7 * self.Z73Li + 4/7 * self.Z74Be + self.Z147N / 2),
		'n_147N': rho * self.Z147N / (14 * const.m_u)
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


	def pp0(self, r_array=None, PP0_array=None, temperature=None, density=None):
		'''
		Method to compute the energy production rate of the PP0 cycle
		'''
		if r_array is None:

			r_array = self.r_PP0

		if PP0_array is None:

			PP0_array = self.PP0

		if temperature is None:

			temperature = self.T

		if density is None:

			density = self.rho

		n_i = self.number_density(density)['n_p']		# Number density of protons
		n_k = n_i
		Q_i , Q_k = self.Q['pp'], self.Q['pd']			# Energy released by fusion of protons to deuterium and deuterium and proton to helium-3
		lmbda = self.reaction_rates(temperature)['pp']	# Reaction rate for two protons

		r_ik = self.r(n_i, n_k, lmbda, density)
		eps = r_ik * (Q_i + Q_k)

		PP0_array[0] = eps
		r_array[0] = r_ik



	def pp1(self, r_array=None, PP1_array=None, temperature=None, density=None):
		'''
		Method to compute the energy production rate of the PP1 cycle
		'''
		if r_array is None:

			r_array = self.r_PP1

		if PP1_array is None:

			PP1_array = self.PP1

		if temperature is None:

			temperature = self.T

		if density is None:

			density = self.rho

		n_i = self.number_density(density)['n_32He']		# Number density of helium-3
		n_k = n_i 
		Q_ik = self.Q['33']									# Energy released by fusing two helium-3 to helium-4
		lmbda = self.reaction_rates(temperature)['33']		# Reaction rate for two helium-3

		r_ik = self.r(n_i, n_k, lmbda, density)
		eps = r_ik * Q_ik

		PP1_array[0] = eps
		r_array[0] = r_ik


	def pp2(self, r_array=None, PP2_array=None, temperature=None, density=None):
		'''
		Method to compute the energy production rate of the PP2 cycle
		'''
		if r_array is None:

			r_array = self.r_PP2

		if PP2_array is None:

			PP2_array = self.PP2 

		if temperature is None:

			temperature = self.T

		if density is None:

			density = self.rho

		lmbda = self.reaction_rates(temperature)
		n = self.number_density(density)

		n_i = np.array([n['n_32He'], n['n_74Be'], n['n_73Li']])				# Number densities of helium-3, beryllium-7, and litium-7
		n_k = np.array([n['n_He'], n['n_e'], n['n_p']])						# Number densities of helium-4, electrones, and protons
		Q_ik = np.array([self.Q['34'], self.Q['e7'], self.Q['17_mark']])	# Energy releases of the fusion reactions 
		lmbda_ik = np.array([lmbda['34'], lmbda['e7'], lmbda['17_mark']])	# Reaction rates for the PP2 branch

		N_A = const.N_A * 1e-6

		'''
		Setting the upper limit for T < 10^6 K
		'''
		if  temperature < 1e6 and Q_ik[1] > 1.57e-7 / (n_k[1] * N_A):

			Q_ik[1] = 1.57e-7 / (n_k[1] * N_A)

		for i in range(3):

			r_ik = self.r(n_i[i], n_k[i], lmbda_ik[i], density)
			eps = r_ik * Q_ik[i]

			PP2_array[i] = eps
			r_array[i] = r_ik

	
	def pp3(self, r_array=None, PP3_array=None, temperature=None, density=None):
		'''
		Method to compute the energy production rate of the PP3 cycle
		'''
		if r_array is None:

			r_array = self.r_PP3

		if PP3_array is None:

			PP3_array = self.PP3

		if temperature is None:

			temperature = self.T 

		if density is None:

			density = self.rho 

		lmbda = self.reaction_rates(temperature)
		n = self.number_density(density)

		n_i = np.array([n['n_32He'], n['n_74Be']])
		n_k = np.array([n['n_He'], n['n_p']])
		Q_ik = np.array([self.Q['34'], self.Q['17']])
		lmbda_ik = np.array([lmbda['34'], lmbda['17']])

		for i in range(2):

			r_ik = self.r(n_i[i], n_k[i], lmbda_ik[i], density)
			Q_ik[i] += (self.Q['8'] + self.Q['8_mark']) * (i == 1)
			eps = r_ik * Q_ik[i]

			PP3_array[i] = eps
			r_array[i] = r_ik


	def limit_production_rate(self, r_PP0=None, r_PP1=None, r_PP2=None, r_PP3=None, PP0_array=None, PP1_array=None, PP2_array=None, PP3_array=None):
		'''
		Method to make sure no step consumes more 
		of an element than the last step produces.
		'''
		if r_PP0 is None:

			r_PP0 = self.r_PP0

		if r_PP1 is None:
		
			r_PP1 = self.r_PP1

		if r_PP2 is None:
		
			r_PP2 = self.r_PP2

		if r_PP3 is None:

			r_PP3 = self.r_PP3		

		if PP0_array is None:

			PP0_array = self.PP0 

		if PP1_array is None:

			PP1_array = self.PP1

		if PP2_array is None:

			PP2_array = self.PP2 

		if PP3_array is None:

			PP3_array = self.PP3

		first_32He = r_PP0[0]								# First production of helium-3
		common_32He = np.array([2 * r_PP1[0], r_PP2[0]])	# First reaction in PP1 requires 2 helium-3 and first of PP2 requires one
		sum_32He = np.sum(common_32He)

		if first_32He <= sum_32He:

			R = first_32He / sum_32He
			PP1_array[0] *= R
			r_PP1[0] *= R
			PP2_array[0] *= R
			r_PP2[0] *= R
			PP3_array[0] *= R
			r_PP3[0] *= R

		first_74Be = r_PP2[0]							# First production of beryllium-7
		common_74Be = np.array([r_PP2[1], r_PP3[1]])	# Second reaction in PP2 and PP3 requires beryllium-7
		sum_74Be = np.sum(common_74Be)

		if first_74Be <= sum_74Be:

			R = first_74Be / sum_74Be
			PP2_array[1] *= R
			r_PP2[1] *= R
			PP3_array[1] *= R
			r_PP3[1] *= R

		first_73Li = r_PP2[1]		# First production of lithium-7
		common_73Li = r_PP2[2]		# Third step of PP2 require lithium-7
		sum_73Li = common_73Li

		if first_73Li <= sum_73Li:

			R = first_73Li / sum_73Li
			PP2_array[2] *= R
			r_PP2[2] *= R

	def cno(self, r_array=None, CNO_array=None, temperature=None, density=None):
		'''
		Method to compute the energy production by 
		the CNO cycle.
		'''
		if r_array is None:

			r_array = self.r_CNO

		if CNO_array is None: 

			CNO_array = self.CNO

		if temperature is None:

			temperature = self.T 

		if density is None: 

			density = self.rho

		n = self.number_density(density)
		lmbda_ik = self.reaction_rates(temperature)['p14']

		n_i = n['n_147N']
		n_k = n['n_p']
		Q_ik = self.Q['CNO']

		r_ik = self.r(n_i, n_k, lmbda_ik, density)
		eps = r_ik * Q_ik

		CNO_array[0] = eps
		r_CNO = r_ik








T_sun = 1.57e7
rho_sun = 1.62e5
a = EnergyProduction(T_sun, rho_sun)
a.pp0()
a.pp1()
a.pp2()
a.pp3()
a.cno()
a.limit_production_rate()
print('Sanity check:')
print(f'PP0:\t{a.PP0[0]*a.rho:.3e}\n')
print(f'PP1:\t{a.PP1[0]*a.rho:.3e}\n')
print(f'PP2:\t{a.PP2[0]*a.rho:.3e}')
print(f'\t{a.PP2[1]*a.rho:.3e}')
print(f'\t{a.PP2[2]*a.rho:.3e}\n')
print(f'PP3:\t{a.PP3[0]*a.rho:.3e}')
print(f'\t{a.PP3[1]*a.rho:.3e}\n')
print(f'CNO:\t{a.CNO[0]*a.rho:.3e}')



