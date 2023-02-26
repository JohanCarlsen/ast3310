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

	########################################################################################################

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

	########################################################################################################

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

	########################################################################################################

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

	########################################################################################################	

	def pp0(self, r_array, PP0_array, temperature, density):
		'''
		Method to compute the energy production rate of the PP0 cycle
		'''
		n_i = self.number_density(density)['n_p']		# Number density of protons
		n_k = n_i
		Q_i , Q_k = self.Q['pp'], self.Q['pd']			# Energy released by fusion of protons to deuterium and deuterium and proton to helium-3
		lmbda = self.reaction_rates(temperature)['pp']	# Reaction rate for two protons

		r_ik = self.r(n_i, n_k, lmbda, density)
		eps = r_ik * (Q_i + Q_k)

		PP0_array[0] = eps
		r_array[0] = r_ik

	########################################################################################################	

	def pp1(self, r_array, PP1_array, temperature, density):
		'''
		Method to compute the energy production rate of the PP1 cycle
		'''
		n_i = self.number_density(density)['n_32He']		# Number density of helium-3
		n_k = n_i 
		Q_ik = self.Q['33']									# Energy released by fusing two helium-3 to helium-4
		lmbda = self.reaction_rates(temperature)['33']		# Reaction rate for two helium-3

		r_ik = self.r(n_i, n_k, lmbda, density)
		eps = r_ik * Q_ik

		PP1_array[0] = eps
		r_array[0] = r_ik

	########################################################################################################	

	def pp2(self, r_array, PP2_array, temperature, density):
		'''
		Method to compute the energy production rate of the PP2 cycle
		'''
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
		if  temperature < 1e6 and lmbda_ik[1] > 1.57e-7 / (n_k[1] * N_A):

			lmbda_ik[1] = 1.57e-7 / (n_k[1] * N_A)

		for i in range(3):

			r_ik = self.r(n_i[i], n_k[i], lmbda_ik[i], density)
			eps = r_ik * Q_ik[i]

			PP2_array[i] = eps
			r_array[i] = r_ik

	########################################################################################################

	def pp3(self, r_array, PP3_array, temperature, density):
		'''
		Method to compute the energy production rate of the PP3 cycle
		'''
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

	########################################################################################################		

	def cno(self, r_array, CNO_array, temperature, density):
		'''
		Method to compute the energy production by 
		the CNO cycle.
		'''

		n = self.number_density(density)
		lmbda_ik = self.reaction_rates(temperature)['p14']

		n_i = n['n_147N']
		n_k = n['n_p']
		Q_ik = self.Q['CNO']

		r_ik = self.r(n_i, n_k, lmbda_ik, density)
		eps = r_ik * Q_ik

		CNO_array[0] = eps
		r_CNO = r_ik

	########################################################################################################		

	def limit_production_rate(self, r_PP0, r_PP1, r_PP2, r_PP3, PP0_array, PP1_array, PP2_array, PP3_array):
		'''
		Method to make sure no step consumes more 
		of an element than the last step produces.
		'''

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

	########################################################################################################	

	def run_all_cycles(self, temperature=None, density=None, PP0=None, PP1=None, PP2=None, PP3=None, CNO=None, r_PP0=None, r_PP1=None, r_PP2=None, r_PP3=None, r_CNO=None):
		'''
		This method runs each of the methods for computing the energy output
		from all of the PP branches and the CNO cycle, before limiting the 
		production rates and updating the energy outputs.
		'''
		if temperature is None:

			temperature = self.T 

		if density is None:

			density = self.rho 

		if PP0 is None:

			PP0 = self.PP0

		if PP1 is None: 

			PP1 = self.PP1 

		if PP2 is None: 

			PP2 = self.PP2 

		if PP3 is None:

			PP3 = self.PP3 

		if CNO is None:

			CNO = self.CNO 

		if r_PP0 is None:

			r_PP0 = self.r_PP0

		if r_PP1 is None:

			r_PP1 = self.r_PP1

		if r_PP2 is None:

			r_PP2 = self.r_PP2

		if r_PP3 is None:

			r_PP3 = self.r_PP3

		if r_CNO is None:

			r_CNO = self.r_CNO

		self.pp0(r_PP0, PP0, temperature, density)
		self.pp1(r_PP1, PP1, temperature, density)
		self.pp2(r_PP2, PP2, temperature, density)
		self.pp3(r_PP3, PP3, temperature, density)
		self.cno(r_CNO, CNO, temperature, density)
		self.limit_production_rate(r_PP0, r_PP1, r_PP2, r_PP3, PP0, PP1, PP2, PP3)

	########################################################################################################	

	def sanity_check(self):
		'''
		Method to compare the methods in the class with
		known values of the Sun, using the temperature
		and density of the Sun, and the Sun's density
		and T=10^8 K.
		'''
		T_sun = 1.57e7
		T8 = 1e8
		rho_sun = 1.62e5

		Sun_test = EnergyProduction(T_sun, rho_sun)
		Sun_test.run_all_cycles()

		T8_test = EnergyProduction(T8, rho_sun)
		T8_test.run_all_cycles()

		'''
		Known values for the Sun
		'''
		Sun_known = np.array([4.04e2, 8.68e-9, 4.86e-5, 1.49e-6, 5.29e-4, 1.63e-6, 9.18e-8])
		T8_known = np.array([7.34e4, 1.09e0, 1.74e4, 1.22e-3, 4.35e-1, 1.26e5, 3.45e4])

		'''
		Creating a dictionary with all the values that will be used
		to print a nicely formatted table.
		'''
		print('\nRUNNING SANITY CHECK\nAll energy units are J s^-1\n')
		print(f"{'':<10} |{'':<5} {'Sun parameters':<31} | {'':<3} {'Sun density, T=10^8 K'}")
		print(f"{'Reaction':<10} | {'Computed':<12} {'Actual':<12} {'Rel. err.':<10} | {'Computed':<12} {'Actual':<12} {'Rel. err.'}")

		table = {
		'PP0': [f'{Sun_test.PP0[0] * rho_sun:.3e}', f'{Sun_known[0]:.3e}', f'{T8_test.PP0[0] * rho_sun:.3e}', f'{T8_known[0]:.3e}'],
		'PP1': [f'{Sun_test.PP1[0] * rho_sun:.3e}', f'{Sun_known[1]:.3e}', f'{T8_test.PP1[0] * rho_sun:.3e}', f'{T8_known[1]:.3e}'],
		'PP2': [f'{Sun_test.PP2[0] * rho_sun:.3e}', f'{Sun_known[2]:.3e}', f'{T8_test.PP2[0] * rho_sun:.3e}', f'{T8_known[2]:.3e}'],
		'   ': [f'{Sun_test.PP2[1] * rho_sun:.3e}', f'{Sun_known[3]:.3e}', f'{T8_test.PP2[1] * rho_sun:.3e}', f'{T8_known[3]:.3e}'],
		'   ': [f'{Sun_test.PP2[2] * rho_sun:.3e}', f'{Sun_known[4]:.3e}', f'{T8_test.PP2[2] * rho_sun:.3e}', f'{T8_known[4]:.3e}'],
		'PP3': [f'{Sun_test.PP3[1] * rho_sun:.3e}', f'{Sun_known[5]:.3e}', f'{T8_test.PP3[1] * rho_sun:.3e}', f'{T8_known[5]:.3e}'],
		'CNO': [f'{Sun_test.CNO[0] * rho_sun:.3e}', f'{Sun_known[6]:.3e}', f'{T8_test.CNO[0] * rho_sun:.3e}', f'{T8_known[6]:.3e}']
		}

		for key, value in table.items():

			sun_test, sun_known, t8_test, t8_known = value
			sun_rel_err = f'{abs(float(sun_test) - float(sun_known)) / abs(float(sun_known)):.3e}'
			t8_rel_err = f'{abs(float(t8_test) - float(t8_known)) / abs(float(t8_known)):.3e}'

			print(f'{key:<10} | {sun_test:<12} {sun_known:<12} {sun_rel_err:<10} | {t8_test:<12} {t8_known:<12} {t8_rel_err}')

		print('\nTEST FINISHED\n')

########################################################################################################
	