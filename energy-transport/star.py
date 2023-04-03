import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp

# Extracting data from opacity.txt.
log_R = np.loadtxt('opacity.txt', usecols=range(1, 20), max_rows=1)		# [gcm^-3 K^-3]
data = np.loadtxt('opacity.txt', skiprows=2)
log_T = data[:, 0]														# [K]
log_K = data[:, 1:]														# [cm^2 g^-1]

spline = RectBivariateSpline(log_T, log_R, log_K)

t = [3.75, 3.755, 3.755, 3.755, 3.755, 3.77, 3.78, 3.795, 3.77, 3.775, 3.780, 3.795, 3.8]
r = [-6, -5.95, -5.8, -5.7, -5.5, -5.95, -5.95, -5.95, -5.8, -5.75, -5.7, -5.55, -5.5]

k = spline.ev(t, r)

print(f"{'log10(T)':<12}{'log10(R) [cgs]':<20}{'log10(K) [cgs]':<20}{'K [SI]'}")

for i in range(len(t)):

	ti = f"{t[i]:.3f}"
	ri = f"{r[i]:.2f}"
	ki = f"{k[i]:.2f}"
	ki_SI = f"{10**k[i] * 1e-1:.2e}"

	print(f"{'':<1}{ti:<15}{ri:<20}{ki:<15}{ki_SI}")


