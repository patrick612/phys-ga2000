#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:31:21 2023

@author: jaewoolee
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
"""
N = 1000
L = 10**(-8)
a = L/N
h = 10**(-18)

a_1 = complex(1, (5.786 * 10**(-6)) * h/(a**2))
a_2 = complex(0, -1 * (5.786 * 10**(-6)) * h/(2 * a**2))
A = np.zeros([N,N]) + 1j*np.zeros([N,N])

for i in range(0, N-1):
    for j in range(0, N-1):
        if A[i][j] != 0:
            continue
        if i == j:
            A[i][j] = a_1
        if j == i-1 or j == i+1:
            A[i][j] = a_2
            
b_1 = complex(1, -1 * (5.786 * 10**(-6)) * h/(a**2))
b_2 = complex(0, (5.786 * 10**(-6)) * h/(2 * a**2))
B = np.zeros([N,N]) + 1j*np.zeros([N,N])

for i in range(0, N-1):
    for j in range(0, N-1):
        if B[i][j] != 0:
            continue
        if i == j:
            B[i][j] = b_1
        if j == i-1 or j == i+1:
            B[i][j] = b_2
            
            


L = 1
x = np.linspace(0, L, 1000)
x0 = L/2
sigma = L/10
kappa = 50/L
psi = np.exp(-1* ((x - x0)**2)/(2*sigma**2)) * np.exp(1j * kappa * x)
psi[0] = psi[-1] = 0

psi_plush = np.linalg.solve(A, np.matmul(B, psi))
psi_plush = scipy.linalg.solve_banded((1,1), A, np.matmul(B,psi))


#lets make banded matrix
A_0 = np.ones(N+1) * a_2
A_1 = np.ones(N+1) * a_1
A = np.array([A_0, A_1, A_0])

v = np.matmul(B,psi)
v = np.append(np.array([0]), v)
psi_plush = scipy.linalg.solve_banded((1,1), A, v)

"""
###################
N = 1000
L = 10**(-8)
a = L/N
h = 10**(-18)

a_1 = 1 + (5.786 * 10**(-6)) * h/(a**2)
a_2 = -1 * (5.786 * 10**(-6)) * h/(2 * a**2)

            
x = np.linspace(0, L, 1001)
x0 = L/2
sigma = L/100
kappa = 500/L
psi = np.exp(-1* ((x - x0)**2)/(2*sigma**2)) * np.exp(1j * kappa * x)
psi[0] = psi[-1] = 0


def evolve_psi(time_step, N, L):
    a = L/N
    h = 10**(-18)
    
    a_1 = 1 + 1j*(5.786 * 10**(-6)) * h/(a**2)
    a_2 = -1 * 1j*(5.786 * 10**(-6)) * h/(2 * a**2)
    
    x = np.linspace(0, L, 1001)
    x0 = L/2
    sigma = L/100
    kappa = 500/L
    psi = np.exp(-1* ((x - x0)**2)/(2*sigma**2)) * np.exp(1j * kappa * x)
    psi[0] = psi[-1] = 0

    A_0 = np.ones(N+1) * a_2
    A_1 = np.ones(N+1) * a_1
    A = np.array([A_0, A_1, A_0])
    
    b_1 = 1 - 1j*(5.786 * 10**(-6)) * h/(a**2)
    b_2 = 1j*(5.786 * 10**(-6)) * h/(2 * a**2)
    B = np.zeros([N+1,N+1]) + 1j*np.zeros([N+1,N+1])

    for i in range(0, N-1):
        for j in range(0, N-1):
            if B[i][j] != 0:
                continue
            if i == j:
                B[i][j] = b_1
            if j == i-1 or j == i+1:
                B[i][j] = b_2
    
    psi_list = []
    for i in range(time_step):
        psi_list.append(psi)
        v = np.matmul(B,psi)
        psi = scipy.linalg.solve_banded((1,1), A, v)
        psi[0] = psi[-1] = 0
    
    return(psi_list)

psi_list = evolve_psi(100000, N, L)

plt.plot(np.linspace(0, 1, 1001), psi_list[0])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 0 s')

plt.plot(np.linspace(0, 1, 1001), psi_list[100])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 1e-16 s')

plt.plot(np.linspace(0, 1, 1001), psi_list[1000])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 1e-15 s')

plt.plot(np.linspace(0, 1, 1001), psi_list[10000])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 1e-14 s')

plt.plot(np.linspace(0, 1, 1001), psi_list[15000])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 1.5e-14 s')


plt.plot(np.linspace(0, 1, 1001), psi_list[20000])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 2e-14 s')

plt.plot(np.linspace(0, 1, 1001), psi_list[30000])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 3e-14 s')

plt.plot(np.linspace(0, 1, 1001), psi_list[50000])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 5e-14 s')

plt.plot(np.linspace(0, 1, 1001), psi_list[90000])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 9e-14 s')

plt.plot(np.linspace(0, 1, 1001), psi_list[99000])
plt.xlabel('Position (1e-8 m)' )
plt.ylabel('Probability density (1/sqrt(m))')
plt.title('Wave function at t = 9.9e-14 s')
        
        
        
        
        
