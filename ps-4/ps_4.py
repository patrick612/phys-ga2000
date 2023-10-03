import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import roots_hermite


#1 The function to calculate gaussian quadrature was taken from the textbook
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

#1a)
def cv(T, N):
    theta_D = 428
    def f(x):
        return np.power((T/theta_D), 3)*np.power(x, 4)*np.exp(x)/(np.power((np.exp(x) - 1), 2))

    x, w = gaussxwab(N, 0, theta_D/T)
    integral = 0.0

    for k in range(N):
        integral += w[k]*f(x[k])
    return integral

#b)
int_list = []
for i in np.arange(5, 500, 1):
    int_val = cv(i, 50) * 9 * 10**(-3) * 6.022*10**(28) * 1.381*10**(-23)
    int_list.append(int_val)


plt.plot(np.arange(5, 500, 1), int_list)
plt.xlabel("Temperature (K)")
plt.ylabel("Specific Heat (J/K)")
plt.title("Heat Capacity")
plt.show()

#c)
N_list = []
for i in np.arange(10, 80, 10):
    int_val = cv(50, i)* 9 * 10**(-3) * 6.022*10**(28) * 1.381*10**(-23)
    N_list.append(int_val)

plt.plot(np.arange(10, 80, 10), N_list)
plt.xlabel("Number of Sampled Points N")
plt.ylabel("Specific Heat at T = 50K (J/K)")
plt.title("Convergence of an Integral with Increase in Sampled Points\n")
plt.show()
#problem 2
def T(a):
    N = 20
    def f(x):
        return 1/(np.sqrt((np.power(a, 4) - np.power(x, 4)))) * np.sqrt(8)
    x, w = gaussxwab(N, 0, a)
    integral = 0.0
    for k in range(N):
        integral += w[k]*f(x[k])
    return integral

int_list = []
for i in np.arange(0.01, 2, 0.01):
    int_val = T(i)
    int_list.append(int_val)

plt.plot(np.arange(0.01, 2, 0.01), int_list)
plt.xlabel("Amplitude (m)")
plt.ylabel("Period of Oscillation (s)")
plt.title("Period of Oscillation of Anharmonic Oscillator")
plt.show()

#3a)

def H(n, x):
    if n==0:
        return np.ones(x.shape)
    elif n==1:
        return 2*x
    else:
        return 2*x*H(n-1, x)-2*(n-1)*H(n-2, x)


def psi(n):
    psi_list = []
    x = np.linspace(-4, 4, 1000)
    H_n = H(n, x)
    psi = 1/np.sqrt(np.power(2, n)*math.factorial(n)*np.sqrt(np.pi)) * np.exp(-1*np.power(x, 2)/2) * H_n

    return psi

plt.plot(np.linspace(-4, 4, 1000), psi(0), label = 'n = 0')
plt.plot(np.linspace(-4, 4, 1000), psi(1), label = 'n = 1')
plt.plot(np.linspace(-4, 4, 1000), psi(2), label = 'n = 2')
plt.plot(np.linspace(-4, 4, 1000), psi(3), label = 'n = 3')

plt.xlabel('Position (sqrt[ħ/(mω)])')
plt.ylabel('Probability Density  (1/sqrt[√ħ/(mω)])')
plt.title('Wave Function of Harmonic Oscillator')
plt.legend()
plt.show()

#3b)
def psi(n):
    psi_list = []
    x = np.linspace(-10, 10, 1000)
    H_n = H(n, x)
    psi = 1/np.sqrt(np.power(2, n)*math.factorial(n)*np.sqrt(np.pi)) * np.exp(-1*np.power(x, 2)/2) * H_n

    return psi

n = 30
psi_list = psi(n)
plt.plot(np.linspace(-10, 10, 1000), psi_list)
plt.xlabel('Position (sqrt[ħ/(mω)])')
plt.ylabel('Probability Density (1/sqrt[√ħ/(mω)])')
plt.title('Wave Function of Harmonic Oscillator n = ' + str(n))
plt.show()

#3c)
def pos_uncertainty(n, N):
    def f(x):
        x_new = x/(1 - np.power(x, 2))
        H_n = H(n, np.array([x_new]))
        return (1 + np.power(x, 2))/np.power((1 - np.power(x, 2)),2) * np.power(x_new, 2) * np.power(np.abs(1/np.sqrt(np.power(2, n)*math.factorial(n)*np.sqrt(np.pi)) * np.exp(-1*np.power(x_new, 2)/2) * H_n[0]), 2)
    x, w = gaussxwab(N, -1, 1)
    integral = 0.0

    for k in range(N):
        integral += w[k]*f(x[k])
    return integral

ps_gq = np.sqrt(pos_uncertainty(5, 100))
#3d)
def pos_uncertainty_gh(n, N):
    def f(x):
        H_n = H(n, np.array([x]))
        return np.power(1/np.sqrt(np.power(2, n)*math.factorial(n)*np.sqrt(np.pi)) * H_n[0], 2) * np.power(x, 2)
    x, w = roots_hermite(N)
    integral = 0.0
    for k in range(N):
        integral += w[k]*f(x[k])
    return integral
ps_gh = np.sqrt(pos_uncertainty_gh(5, 100))