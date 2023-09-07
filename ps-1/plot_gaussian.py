import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian():
    x_var = np.array([])
    y_var = np.array([])
    for i in np.linspace(-10,10,1000):
        x_var = np.append(x_var,i)
        y = 1/(np.sqrt(2*np.pi)*3)*np.exp(-0.5*(i/3)**2)
        y_var = np.append(y_var,y)
    return x_var, y_var


x_ax, y_ax = plot_gaussian()

plt.plot(x_ax, y_ax)
plt.xlabel('State')
plt.ylabel('Probability')
plt.title('Normalized Gaussian Distribution')
plt.savefig('Gaussian Plot')
plt.show
