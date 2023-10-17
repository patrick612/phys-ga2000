import numpy as np
import matplotlib.pyplot as plt
import math
import os
import numpy.linalg as linalg

#Problem 1 a)
import scipy.integrate


def integrand_plot():
    def func(a,x):
        return np.power(x, (a-1)) * np.power(np.e, -x)
    a_list = [2,3,4]
    for i in a_list:
        xval = np.linspace(0, 5, 250)
        yval = func(i, xval)
        plt.plot(xval, yval, label = str(i))
    plt.legend()
    plt.xlabel('Position')
    plt.ylabel('Integrand')
    plt.title('Integrand of Gamma Function')
    plt.show()

#Problem 1 d)
def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def gamma(a):
    c = a - 1
    def func(a,x):
        c = a - 1
        z = x / (c + x)
        dx = np.power((c + x), 2)/c
        return np.power(np.e, (a-1)*np.log(z)) * np.power(np.e, -z) * dx
    N = 30
    x, w = gaussxwab(N, 0, c)
    return np.sum(w * func(a,x))

def gamma(a):
    c = a - 1
    def func(a,x):
        c = a - 1
        x_new = x * c / (1 - x)
        return np.power(np.e, (a-1)*np.log(x_new) - x_new) * c/(np.power((1-x), 2))
    N = 30
    x, w = gaussxwab(N, 0, 1)
    return np.sum(w * func(a,x))

gamma_list = []
factorial_list = []
for i in np.arange(3, 11, 1):
    gam = gamma(i)
    gamma_list.append(gam)
    factorial_list.append(math.factorial(i))


#Problem 2
#a)
def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

data = []
with open(os.path.expanduser("~/Desktop/NYU_Courses/GA2000/phys-ga2000/ps-5/signal.dat"), 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split('|')
        for i in k:
            if is_float(i):
                data.append(float(i))

data = np.array(data, dtype='float')
time = data[::2]/np.max(data[::2])
signal = data[1::2]

plt.scatter(time, signal)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Raw Data')

#b)
A = np.zeros((len(time), 4))
A[:, 0] = 1.
A[:, 1] = time
A[:, 2] = time**2
A[:, 3] = time**3

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
print(c)

predicted = A.dot(c)
plt.scatter(time, predicted, label = 'Fitted Curve')
plt.scatter(time, signal, label = 'Raw Data')
plt.xlabel('Time (10e8 s)')
plt.ylabel('Signal')
plt.legend(['Fitted Curve', 'Raw Data'])
plt.title('Fitting Raw Data using Sinusoidal Basis')

#c)
A = np.zeros((len(time), 4))
A[:, 0] = 1.
A[:, 1] = time
A[:, 2] = time**2
A[:, 3] = time**3

predicted = A.dot(c)
residuals = signal - predicted

plt.scatter(time, residuals)
plt.xlabel('Time (10e8 s)')
plt.ylabel('Residuals')
plt.title('Residuals SVD Fitting')

#d)
def poly_svd(N, time):
    A = np.zeros((len(time), N+1))
    for i in range(0, N+1):
        A[:, i] = time**i

    (u, w, vt) = np.linalg.svd(A, full_matrices=False)
    ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
    c = ainv.dot(signal)
    return A, w, c

N = 500
A, w, c = poly_svd(N, time)
predicted = A.dot(c)
plt.scatter(time, predicted, label = 'Fitted Curve')
plt.scatter(time, signal, label = 'Raw Data')
plt.xlabel('Time (10e8 s)')
plt.ylabel('Signal')
plt.legend(['Fitted Curve', 'Raw Data'])
plt.title('Fitting Raw Data using ' + str(N) + 'th Order Sinusoidal Basis')

print('condition number is ', np.max(w)/np.min(w))

condition_number_list = []
pol_deg = 500
for i in range(0, pol_deg, 2):
    A, w, c = poly_svd(i, time)
    condition_number_list.append(np.max(w)/np.min(w))


plt.plot(range(0, pol_deg, 2), condition_number_list)
#plt.ylim(np.max(condition_number_list), np.min(condition_number_list))
plt.xlabel('Polynomial Degree')
plt.ylabel('Condition Number')
plt.title('Condition Number for Polynomial Degrees')

#e)
half_time = (np.max(time) - np.min(time))/2
omega = 2 * np.pi * 1/half_time

A = np.zeros((len(time), 23))
A[:, 0] = 1.
A[:, 1] = np.cos(omega*time)
A[:, 2] = np.sin(omega*time)
A[:, 3] = np.cos(2*omega*time)
A[:, 4] = np.sin(2*omega*time)
A[:, 5] = np.cos(3*omega*time)
A[:, 6] = np.sin(3*omega*time)
A[:, 7] = np.cos(4*omega*time)
A[:, 8] = np.sin(4*omega*time)
A[:, 9] = np.cos(5*omega*time)
A[:, 10] = np.sin(5*omega*time)
A[:, 11] = np.cos(6*omega*time)
A[:, 12] = np.sin(6*omega*time)
A[:, 13] = np.cos(7*omega*time)
A[:, 14] = np.sin(7*omega*time)
A[:, 15] = np.cos(8*omega*time)
A[:, 16] = np.sin(8*omega*time)
A[:, 17] = np.cos(9*omega*time)
A[:, 18] = np.sin(9*omega*time)
A[:, 19] = np.cos(10*omega*time)
A[:, 20] = np.sin(10*omega*time)
A[:, 21] = np.cos(11*omega*time)
A[:, 22] = np.sin(11*omega*time)

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
print(c)

predicted = A.dot(c)
residuals = signal - predicted

plt.scatter(time, predicted, label = 'Fitted Curve')
plt.scatter(time, signal, label = 'Raw Data')
plt.xlabel('Time (10e8 s)')
plt.ylabel('Signal')
plt.legend(['Fitted Curve', 'Raw Data'])
plt.title('Fitting Raw Data using Sinusoidal Basis')

print('period is ', 0.5/(np.max(time)-np.min(time)))
print('condition number is ', np.max(w)/np.min(w))