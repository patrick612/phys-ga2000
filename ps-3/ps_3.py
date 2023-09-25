import numpy as np
import matplotlib.pyplot as plt
import scipy
import timeit
import time
from random import random


#1 (4.2 in Newmann)
#a)
def func(x):
    f_x = x*(x-1)
    return f_x

def deriv(point, delta):
    deriv_point = (func(point + delta) - func(point))/delta
    return deriv_point

delta1 = 1/np.power(10,2)
deriv1 = deriv(1, delta1)
print(deriv1)

#b)
power_list = []
for i in np.arange(4, 16, 2):
    power_list.append(1/np.power(10, i))

deriv_list = []
for i in np.arange(0, len(power_list), 1):
    derivative = deriv(1, power_list[i])
    deriv_list.append(derivative)


#2
N_array = np.arange(10, 200, 10)
time_array = []

def for_loop_mult(N, A, B):
    C = np.zeros([N,N], float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += A[i,k]*B[k,j]
    return C

def dot_mult(A, B):
    C = np.dot(A,B)
    return C

def operation_time(N, A, B, dot = False):
    if dot == False:
        t1 = time.time()
        a= for_loop_mult(N, A, B)
        t2 = time.time()
    else:
        t1 = time.time()
        a = dot_mult(A,B)
        t2 = time.time()
    t = t2-t1
    return t


t_list_forloop = []
t_list_dot = []
for ind in np.arange(0, len(N_array), 1):
    N = N_array[ind]
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    t_for = operation_time(N, A, B)
    t_dot = operation_time(N, A, B, dot=True)
    t_list_forloop.append(t_for)
    t_list_dot.append(t_dot)

N_cubed = []
for i in np.arange(0, len(N_array), 1):
    n_cubed = 0.5*np.power(N_array[i], 3)/np.power(10,6)
    N_cubed.append(n_cubed)

plt.plot(N_array, t_list_forloop, label='computation time')
plt.plot(N_array, N_cubed, label='1/2 * 10E-6 * NE3')
plt.xlabel('Array Length')
plt.ylabel('Computation Time')
plt.title('For Loop Computation Time Scaling')
plt.legend()

plt.plot(N_array, t_list_dot)
plt.xlabel('Array Length')
plt.ylabel('Computation Time')
plt.title('Dot Product Computation Time Scaling')


# check if computation time scales like N^3 for for loop method
plt.plot(N_array, N_cubed)

#Problem 3
#a)
bi_213 = 10000
bi_209 = 0
pb = 0
tl = 0
isotope_nums = {'bi_213': bi_213, 'bi_209': bi_209, 'pb':pb, 'tl':tl}
def pb_decay(start_time, cur_time, isotope_nums):
    t = cur_time - start_time
    prob = 1 - 1 / (np.power(2, t / (198)))
    if random() < prob:
        isotope_nums['pb'] = isotope_nums['pb'] -1
        isotope_nums['bi_209'] = isotope_nums['bi_209'] +1
        return 'bi_209', isotope_nums
    else:
        return False, isotope_nums

#b)
def tl_decay(start_time, cur_time, isotope_nums):
    t = cur_time - start_time
    prob = 1 - 1 / (np.power(2, t / (132)))
    if random() < prob:
        isotope_nums['tl'] = isotope_nums['tl'] - 1
        isotope_nums['pb'] = isotope_nums['pb'] + 1
        return 'pb', isotope_nums
    else:
        return False, isotope_nums

#c)
def bi_decay(start_time, cur_time, isotope_nums):
    t = cur_time - start_time
    prob = 1 - 1 / (np.power(2, t / (2760)))
    if random() < prob:
        if random() < 0.9791:
            isotope_nums['bi_213'] = isotope_nums['bi_213'] - 1
            isotope_nums['pb'] = isotope_nums['pb'] + 1
            return 'pb', isotope_nums
        else:
            isotope_nums['bi_213'] = isotope_nums['bi_213'] - 1
            isotope_nums['tl'] = isotope_nums['tl'] + 1
            return 'tl', isotope_nums
    else:
        return False, isotope_nums



def decay(tot_time, tot_isotopes):
    cur_time = 1
    bi_213 = tot_isotopes
    bi_209 = 0
    pb = 0
    tl = 0
    bi_213_list = np.ones(tot_isotopes)
    bi_209_list = np.zeros(tot_isotopes)
    pb_list = np.zeros(tot_isotopes)
    tl_list = np.zeros(tot_isotopes)
    iso_num_list = [[bi_213, bi_209, pb, tl]]
    isotope_nums = {'bi_213': bi_213, 'bi_209': bi_209, 'pb': pb, 'tl': tl}
    for i in np.arange(1, tot_time+1, 1):
        for pb in np.arange(0, len(pb_list), 1):
            if pb_list[pb] == 0:
                pass
            else:
                pbdecay, isotope_nums = pb_decay(pb_list[pb], cur_time, isotope_nums)
                if pbdecay == 'bi_209':
                    pb_list[pb] = 0
                    bi_209_list[pb] = cur_time
                else:
                    pb_list[pb] = cur_time

        for tl in np.arange(0, len(tl_list), 1):
            if tl_list[tl] == 0:
                pass
            else:
                tldecay, isotope_nums = tl_decay(tl_list[tl], cur_time, isotope_nums)
                if tldecay == 'pb':
                    tl_list[tl] = 0
                    pb_list[tl] = cur_time
                else:
                    tl_list[tl] = cur_time

        for bi213 in np.arange(0, len(bi_213_list), 1):
            if bi_213_list[bi213] == 0:
                pass
            else:
                bi213decay, isotope_nums = bi_decay(bi_213_list[bi213] ,cur_time, isotope_nums)
                if bi213decay == 'pb':
                    bi_213_list[bi213] = 0
                    pb_list[bi213] = cur_time
                elif bi213 == 'tl':
                    bi_213_list[bi213] = 0
                    tl_list[bi213] = cur_time
                else:
                    bi_213_list[bi213] = cur_time
        cur_time += 1
        iso_num_list.append([isotope_nums['bi_213'],isotope_nums['bi_209'], isotope_nums['pb'], isotope_nums['tl']])
    return iso_num_list

decay_list = decay(20000, 10000)
x_axis = np.arange(0, 20001, 1)
bi_213 = []
bi_209 = []
pb = []
tl = []
for i in np.arange(0, len(decay_list), 1):
    bi_213.append(decay_list[i][0])
    bi_209.append((decay_list[i][1]))
    pb.append(decay_list[i][2])
    tl.append(decay_list[i][3])

plt.plot(x_axis, bi_213, label='Bi 213')
plt.plot(x_axis, bi_209, label='Bi 209')
plt.plot(x_axis, pb, label='Pb')
plt.plot(x_axis, tl, label='Tl')
plt.xlabel('Time')
plt.ylabel('Isotope Number')
plt.title('Isotope Decay')
plt.legend()

#problem 4
time_list = []
def prb_dist():
    rand_num = random()
    num = -183.18/np.log(2) * np.log(1-rand_num)
    return num

for i in np.arange(1000):
    t = prb_dist()
    time_list.append(t)

tlist = np.array(time_list)
tlist = np.sort(tlist)

not_decay_num_list = []
for t in np.arange(0, 1000, 1):
    not_decay = []
    not_decay = tlist < t
    not_decay_num = np.sum(not_decay)
    not_decay_num_list.append(not_decay_num)

plt.plot(np.arange(0, 1000, 1), not_decay_num_list)
plt.xlabel('Time')
plt.ylabel('Number of isotopes that have not decayed')
plt.title('Tl 208 Decay')


