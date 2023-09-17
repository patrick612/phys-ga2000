import numpy as np
import struct
import timeit
import matplotlib.pyplot as plt

def quadratic(a, b, c, method_b = False):
    x_a_1 = (-b + np.sqrt(np.power(b, 2) - 4*a*c))/(2*a)
    x_a_2 = (-b - np.sqrt(np.power(b, 2) - 4*a*c))/(2*a)
    x_b_1 = (2*c)/(-b - np.sqrt(np.power(b, 2) - 4*a*c))
    x_b_2 = (2*c)/(-b + np.sqrt(np.power(b, 2) - 4*a*c))

    if method_b == True:
        return x_b_1, x_b_2
    else:
        return x_a_1, x_a_2
