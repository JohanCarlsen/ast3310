import numpy as np 
import astropy.constants as const 

G = const.G
R_sun = const.R_sun 
M_sun = const.M_sun

a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

two_step_a = np.roll(a, 1, axis=0)
print(a)
print(two_step_a)

b = a
print(b)
b[:, 0] = 0 
print(b)