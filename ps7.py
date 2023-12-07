#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:40:13 2023

@author: jaewoolee
"""

#problem 1

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import csv
import os


def s_quad_interp(f, a, b, c):
    """
    inverse quadratic interpolation
    """
    epsilon = 1e-7 #for numerical stability
    s0 = a*f(b)*f(c) / (epsilon + (f(a)-f(b))*(f(a)-f(c)))
    s1 = b*f(a)*f(c) / (epsilon + (f(b)-f(a))*(f(b)-f(c)))
    s2 = c*f(a)*f(b) / (epsilon + (f(c)-f(a))*(f(c)-f(b)))
    return s0+s1+s2

def golden(f, x1, x4):
    z =(1 + np.sqrt(5))/2
    p = x1 * (z-1)
    x2 = (x4 + x1 + p)/(1 + z)
    x3 = (z * x2) - p
    if f(x2) <= f(x3):
        step_size = x4 - x3
        return x1, x2, x3, step_size
    if f(x3) <= f(x2):
        step_size = x2 - x1
        return x2, x3, x4, step_size


def brent(f, a, c, delta = 1e-7):
    a, b, c, step_size = golden(f, a, c)
    while abs(a-c) > delta:
        s = s_quad_interp(f, a,b,c)
     
        if c > a and s > c or s < a:
            a , b, c, step_size = golden(f, a, c)
            continue
        else:
            if f(c) <= f(a):
                step_size2 = c - s
                
                if step_size2 < step_size:
                    a , b, c, step_size = golden(f, a, c)
                    continue
                else:
                    if s > b:
                        a, b, c = b, s, c
                        continue
                    else:
                        a, b, c = s, b, c
                        continue
                        
            if f(a) <= f(c):
                step_size2 = s - a
                
                if step_size2 < step_size:
                    a , b, c, step_size = golden(f, a, c)
                    continue
                else:
                    if s > b:
                        a, b, c = a, b, s
                        continue
                    else:
                        a, b, c = a, s, b
                        continue
        
    return b

def y(x):
    return (x- 0.3)**2 * np.exp(x)

minima = brent(y, -1, 2)
minima_sp = optimize.brent(y, brack=(-1, 2))

minima2 = brent(y, -10, 4)
minima_sp2 = optimize.brent(y, brack=(-10, 4))




#problem 2
file = open(os.path.expanduser("~/Desktop/NYU_Courses/GA2000/phys-ga2000/ps-7/survey.csv"))
csv_file = csv.reader(file)

age = []
ans = []
for i in csv_file:
    age.append(i[0])
    ans.append(i[1])

age.pop(0)
ans.pop(0)
age = np.array(age)
ans = np.array(ans)
age = age.astype(float)
ans = ans.astype(float)

x_sort = np.argsort(age)
age = age[x_sort]
ans = ans[x_sort]


def p(x, beta_0, beta_1):
    return 1/(1+np.exp(-(beta_0+beta_1*x)))

x = 50

beta_0 = np.linspace(-4, 4, 100)
beta_1 = np.linspace(-4, 4, 100)
beta = np.meshgrid(beta_0, beta_1)
p_grid = p(x, *beta)
plt.pcolormesh(*beta, p_grid)
plt.colorbar()
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$p(y_i|x_i=50,\beta_0, \beta_1)$', fontsize = 16)


def log_likelihood(xs, ys, beta):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-p(xs[i], beta_0, beta_1)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return -ll # return log likelihood

ll = log_likelihood(age, ans, beta)
plt.pcolormesh(*beta, ll, shading = 'auto')
plt.colorbar()
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$\mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)


#Find optimal parameters, Covariance Matrix and Errors
b = np.linspace(-4, 4, 100)
def p1(x, b):
    return 1/(1+np.exp(-(b[0]+b[1]*x)))

err_func = lambda b, age, ans: p1(age, b) - ans


# Covariance matrix of parameters
def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance

#Error of parameters
def error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt( np.diag( covariance ))

result = optimize.minimize(lambda b, age,ans: log_likelihood(age,ans, b), [0,0],  args=(age, ans))
hess_inv = result.hess_inv # inverse of hessian matrix
var = result.fun/(len(ans)-2)
dFit = error( hess_inv,  var)
print('Optimal parameters and error:\n\tp: ' , result.x, '\n\tdp: ', dFit)
print('Covariance matrix of optimal parameters:\n\tC: ' , Covariance( hess_inv,  var))

b_0 = result.x[0]
b_1 = result.x[1]



# plot the true probability distribution
age_gap = 3
age_group = np.arange(5, 100, age_gap)
ans_yes = np.zeros(len(age_group))
ans_no = np.zeros(len(age_group))
for i in range(len(age)):
    for j in range(len(age_group)):
        if age_group[j] - age_gap <= age[i] <= age_group[j]:
            if ans[i] == 1:
                ans_yes[j] = ans_yes[j]+1
                
            else:
                ans_no[j] = ans_no[j] + 1
                break
            
            

ans_prob = []
for i in range(len(ans_yes)):
    prob = ans_yes[i]/(ans_yes[i]+ans_no[i])
    ans_prob.append(prob)
    
plt.bar(age_group, ans_prob)

# plot fitted probaility distribution
ans_prob_fitted = []
ages = np.arange(0, 90, 0.1)
for i in range(len(ages)):
    prob = p(ages[i], b_0, b_1)
    ans_prob_fitted.append(prob)
    
    
plt.plot(ages, ans_prob_fitted, label = 'Fitted Curve')
plt.bar(age_group, ans_prob, label ='True Data')
plt.xlabel('Age (Years)')
plt.ylabel('Probability')
plt.title('Probability of Knowing the Phrase')
plt.legend()

plt.plot(ages, ans_prob_fitted, label = 'Fitted Curve')
plt.scatter(age, ans, label ='True Data')
plt.xlabel('Age (Years)')
plt.ylabel('Probability')
plt.title('Probability of Knowing the Phrase')
plt.legend()



#find the gradient
grad_ll_arr = np.gradient(ll)

plt.pcolormesh(*beta, grad_ll_arr[0])
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$\frac{\partial}{\partial \beta_0} \mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)
plt.colorbar()


plt.pcolormesh(*beta, grad_ll_arr[1])
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$\frac{\partial}{\partial \beta_1} \mathcal{L}(\beta_0, \beta_1)$', fontsize = 16)
plt.colorbar()


def hessian(x):
    """
    https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

hess_ll = hessian(ll)

plt.pcolormesh(*beta, hess_ll[0, 0, :, :])
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$\frac{\partial^2}{\partial \beta_0^2} \mathcal{L}(x|\beta_0, \beta_1, \{y_i\})$', fontsize = 16)
plt.colorbar()


plt.pcolormesh(*beta, hess_ll[1, 1, :, :])
plt.xlabel(r'$\beta_0$', fontsize = 16)
plt.ylabel(r'$\beta_1$', fontsize = 16)
plt.title(r'$\frac{\partial^2}{\partial \beta_1^2} \mathcal{L}(x|\beta_0, \beta_1, \{y_i\})$', fontsize = 16)
plt.colorbar()

# Determine the covariance matrix and error
beta_0 = np.linspace(-.1, .1, 100)
beta_1 = np.linspace(-.1, .1, 100)
beta = np.meshgrid(beta_0, beta_1)

ll = log_likelihood(age, ans, beta)
hess_ll = hessian(ll)
cov_mat = np.linalg.inv(hess_ll[0][1])

error_list = []
for i in range(len(cov_mat[0])):
    error = np.sqrt(cov_mat[i][i])
    error_list.append(error)

