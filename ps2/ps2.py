import numpy as np
import struct
import timeit
import matplotlib.pyplot as plt
import cmath

#Problem 1
numb = 100.98763
np_numb = np.float32(numb)
np_b_string = struct.pack('!f', np_numb)
np_bnry = ''.join(format(i, '08b') for i in np_b_string)

difference = 100.98763-np.float32(100.98763)
print("difference of 32bit float representation to actual value is: ", difference)

#Problem 2

#function using for loop
def madelung_for(l, a):
    inv_dist = 0
    for i in np.arange(-l, l+1, 1):
        for j in np.arange(-l, l+1, 1):
            for k in np.arange(-l, l+1, 1):
                if np.abs(i) + np.abs(j) + np.abs(k) == 0:
                    pass
                else:
                    inv_dist = inv_dist + 1/(np.sqrt(i**2 + j**2 + k**2))

    e_charge = 1.60217663 * 10 ** (-19)
    eps = 8.85418782 * 10 ** (-12)
    const = 4 * np.pi * eps * a
    v_total = e_charge/const * inv_dist

    return inv_dist


#function using numpy arrays
def madelung_vect(l,a):
    x_ax = np.arange(-l, l+1, 1)
    y_ax = np.arange(-l, l+1, 1)
    z_ax = np.arange(-l, l+1, 1)
    xv, yv, zv = np.meshgrid(x_ax, y_ax, z_ax, indexing='xy')
    xv  =xv**2
    yv = yv**2
    zv = zv**2
    dist_sqrd = xv+yv+zv
    dist = np.sqrt(dist_sqrd)
    dist[l,l,l] = 1
    inv_dist = 1/dist
    inv_dist[l,l,l] = 0
    inv_dist = np.sum(inv_dist)

    e_charge = 1.60217663 * 10 ** (-19)
    eps = 8.85418782 * 10 ** (-12)
    const = 4 * np.pi * eps * a
    v_total = e_charge / const * inv_dist
    return inv_dist

#function runtime comparison
test_for = madelung_for(10,1)
test_vect = madelung_vect(10, 1)
time_for = timeit.timeit('madelung_for(10,1)', 'from __main__ import madelung_for', number=1000)
time_vect = timeit.timeit('madelung_vect(10,1)', 'from __main__ import madelung_vect', number=1000)

time_diff = time_for/1000 - time_vect/1000
print("run time difference for 1000 runs is: ", time_diff)
print("numpy array method is faster by a factor of ", time_for/time_vect)


#Problem 3
def mandelbrot(iteration, c):
    z = 0
    for i in np.arange(0, iteration, 1):
        if abs(z) > 2:
            return False
        else:
            z = np.power(z, 2) + c
    return True

N = 300
cont_mat = np.zeros((N,N))

for x_it in np.arange(0, N, 1):
    x_ax = np.linspace(-2, 2, N)
    re = x_ax[x_it]
    for y_it in np.arange(0, N, 1):
        y_ax = np.linspace(-2, 2, N)
        im = y_ax[y_it]
        cont_mat[x_it][y_it] = mandelbrot(100, complex(re, im))


cf = plt.pcolormesh(np.linspace(-2, 2, N), np.linspace(-2, 2, N), cont_mat)
plt.title('Mandelbrot')
plt.xlabel('Im(c)')
plt.ylabel('Re(c)')

