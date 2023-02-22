'''
In this program we calculate the energy production in a star, given its 
temperature and density.
'''
import numpy as np 
import scipy.constants as const 
from scipy.integrate import quad 

class EnergyProduction:

	def __init__(self, temperature, density):

		self.T = temperature
		self.T9 = temperature * 1e-9
		self.T9_star = self.T9 / (1 + 4.95e-2 * self.T9)
		self.T9_doublestar = self.T9 / (1 + .759 * self.T9)
		self.rho = density

		self.N_A = const.N_A * 1e6

		self.X = .7; self.Y = .29
		self.Z73Li = 1e-7; self.Z74Be = 1e-7
		self.Y32He = 1e-10; self.Z147N = 1e-11

		self.n_p = self.rho * self.X / const.m_u
		self.n_He = self.rho * self.Y / (4 * const.m_u)
		self.n_32He = self.rho * self.Y32He / (3 * const.m_u)
		self.n_73Li = self.rho * self.Z73Li / (7 * const.m_u)
		self.n_74Be = self.rho * self.Z74Be / (7 * const.m_u)
		self.n_e = self.rho / (2 * const.m_u) * (1 + self.X)

		self.lmbda = {
		'pp': (4.01e-15 * self.T9**(-2/3) * np.exp(-3.38 * self.T9**(-1/3)) * (1 + .123 * self.T9**(1/3) + 1.09 * self.T9**(2/3) + .938 * self.T9)) / self.N_A,
		'33': (6.04e10 * self.T9**(-2/3) * np.exp(-12.276 * self.T9**(-1/3)) * (1 + .034 * self.T9**(1/3) - .522 * self.T9**(2/3) - .124 * self.T9 + .353 * self.T9**(4/3) + .213 * self.T9**(5/3))) / self.N_A,
		'34': (5.61e6 * self.T9_star**(5/6) * self.T9**(-2/3) * np.exp(-12.826 * self.T9_star**(-1/3))) / self.N_A,
		'e7': (1.34e-10 * self.T9**(-1/2) * (1 - .537 * self.T9**(1/3) + 3.86 * self.T9**(2/3) + .0027 * self.T9**(-1) * np.exp(2.515e-3 * self.T9**(-1)))) / self.N_A,
		'17_mark': (1.096e9 * self.T9**(-2/3) * np.exp(-8.472 * self.T9**(-1/3)) - 4.83e8 * self.T9_doublestar**(5/6) * self.T9**(-3/2) * np.exp(-8.472 * self.T9_doublestar**(-1/3)) + 1.06e10 * self.T9**(-3/2) * np.exp(-30.442 * self.T9**(-1))) / self.N_A,
		'17': (3.11e5 * self.T9**(-2/3) * np.exp(-10.262 * self.T9**(-1/3)) + 2.53e3 * self.T9**(-2/3) * np.exp(-7.306 * self.T9**(-1))) / self.N_A,
		'p14': (4.9e7 * self.T9**(-2/3) * np.exp(-15.228 * self.T9**(-1/3) - .092 * self.T9**2) * (1 + .027 * self.T9**(1/3) - .778 * self.T9**(2/3) - .149 * self.T9 + .261 * self.T9**(4/3) + .127 * self.T9**(5/3)) + 2.37e3 * self.T9**(-3/2) * np.exp(-3.011 * self.T9**(-1)) + 2.19e4 * np.exp(-12.53 * self.T9**(-1))) / self.N_A
		}

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


	def r(self, n_i, n_k, lmbda):

		if n_i == n_k:

			delta = 1. 

		else:

			delta = 0

		rate = n_i * n_k * lmbda / (self.rho * (1 + delta))

		return rate 


	def pp0(self):

		n_i = self.n_p; n_k = n_i
		Q_i = self.Q['pp']; Q_k = self.Q['pd']
		lmbda = self.lmbda['pp']

		r_ik = self.r(n_i, n_k, lmbda)
		eps = r_ik * (Q_i + Q_k)

		return eps


	def pp1(self):

		pp0 = self.pp0()
		n_i = self.n_32He; n_k = n_i 
		Q_ik = self.Q['33']
		lmbda = self.lmbda['33']

		r_ik = self.r(n_i, n_k, lmbda)
		eps = r_ik * Q_ik	

		return eps + pp0

	def pp2(self):

		pp0 = self.pp0()
		n = np.array([self.n_32He, self.n_74Be, self.n_73Li], [self.n_He, self.n_e, self.n_p])
		Q = np.array([self.Q['34'], self.Q['e7'], self.Q['17_mark']])
		lmbda = np.array([self.lmbda['34'], self.lmbda['e7'], self.lmbda['17_mark']])

		eps = 0 

		for i in range(3):

			n_i = n[0][i]; n_k = n[1][i]
			Q_ik = Q[i]
			lmbda = lmbda[i]

			r_ik = self.r(n_i, n_k, lmbda)
			eps += r_ik * Q_ik

		return eps + pp0




a = EnergyProduction(1.57e7, 1.62e5)

print(a.pp0())

		
