import numpy as np
import matplotlib.pyplot as plt
import math
import os
import astropy
from astropy.io import fits
from numpy.linalg import eig

#1a)
hdu_list = astropy.io.fits.open(os.path.expanduser("~/Desktop/NYU_Courses/GA2000/phys-ga2000/ps-6/specgrid.fits"))
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

#plotting data corresponding to couple galaxies
plt.plot(logwave, flux[0])
plt.plot(logwave, flux[1])
plt.plot(logwave, flux[2])
plt.xlabel(r'Wavelength $(log_{10}(Angstrom))$')
plt.ylabel('Flux ($10^{-17} erg s^{-1} cm^{-2} A^{-1}$)')
plt.title('Optical Spectra')

plt.plot(logwave, flux[15])
plt.xlabel(r'Wavelength $(log_{10}(Angstrom))$')
plt.ylabel('Flux ($10^{-17} erg s^{-1} cm^{-2} A^{-1}$)')
plt.title('Optical Spectra')

#1b)
normalization = []
for i in range(0, np.shape(flux)[0]):
    norm = np.sum(flux[i])
    flux[i] = flux[i]/norm
    normalization.append(norm)
normalization = np.array(normalization)
plt.plot(np.sum(flux, axis = 1))
plt.ylim(0,2)
plt.xlabel('Galaxy')
plt.ylabel('Sum of spectrum')

#1c)
means = []
for i in range(0, np.shape(flux)[0]):
    mean = np.mean(flux[i])
    flux[i] = flux[i] - mean
    means.append(mean)
means = np.array(means)

plt.plot(logwave, flux[0,:])
plt.xlabel(r'Wavelength $(log_{10}(Angstrom))$')
plt.ylabel('Flux ($10^{-17} erg s^{-1} cm^{-2} A^{-1}$)')
plt.title('Normalized, Residual Optical Spectra')


#1d)
#To confirm that we have found the covariance matrix we could do the following and compare
cov_mat = flux.transpose().dot(flux)
a, v = eig(cov_mat)
plt.plot(range(len(v[0])), v[0], label='0')
plt.plot(range(len(v[0])), v[1], label = '1')
plt.plot(range(len(v[0])), v[2], label = '2')
plt.plot(range(len(v[0])), v[3], label = '3')
plt.plot(range(len(v[0])), v[5], label = '4')
plt.xlabel('Eigen Basis')
plt.ylabel('Coordinates of Eigenvector')
plt.title('Eigen Values of Covariance Matrix')
plt.legend()



#1e)

(u, w, vt) = np.linalg.svd(flux, full_matrices=False)
#plt.plot(range(1,6), a[0:5])
plt.plot(range(1,6), w[a:5])
plt.xlabel('Eigen Vector')
plt.ylabel('Eigen Value')
plt.title('Eigen Values of Covariance Matrix')

#lets see the eigein vectors
plt.plot(range(len(vt[0])), vt.transpose()[0], label='0')
plt.plot(range(len(vt[0])), vt.transpose()[1], label = '1')
plt.plot(range(len(vt[0])), vt.transpose()[2], label = '2')
plt.plot(range(len(vt[0])), vt.transpose()[3], label = '3')
plt.plot(range(len(vt[0])), vt.transpose()[5], label = '4')
plt.xlabel('Eigen Basis')
plt.ylabel('Coordinates of Eigenvector')
plt.title('Eigen Vector using SVD')
plt.legend()

plt.scatter(v[:,0], vt.transpose()[:, 0], label='0')
plt.scatter(v[:,1], vt.transpose()[:, 1], label='1')
plt.scatter(v[:,2], vt.transpose()[:, 2], label='2')
plt.scatter(v[:,3], vt.transpose()[:, 3], label='3')
plt.scatter(v[:,4], vt.transpose()[:, 4], label='4')
plt.xlabel('Covariance Matrix eigenvector')
plt.ylabel('SVD eigenvector')
plt.title('Covariance Matrix eignvector vs SVD Eigenvector')

#1f)
svd_condition_num = np.max(w)/np.min(w)
cm_condition_num = np.max(a)/np.min(a)
print(svd_condition_num)
print(svd_condition_num)

#1g)

def sorted_eigs(r, return_eigvalues = False):
    """
    Calculate the eigenvectors and eigenvalues of the correlation matrix of r
    -----------------------------------------------------
    """
    corr=r.T@r
    eigs=np.linalg.eig(corr) #calculate eigenvectors and values of original 
    arg=np.argsort(eigs[0])[::-1] #get indices for sorted eigenvalues
    eigvec=eigs[1][:,arg] #sort eigenvectors
    eig = eigs[0][arg] # sort eigenvalues
    if return_eigvalues == True:
        return eig, eigvec
    else:
        return eigvec
        
    
def PCA(l, r, project = True):
    """
    Perform PCA dimensionality reduction
    --------------------------------------------------------------------------------------
    """
    eigvector = sorted_eigs(r)
    eigvec=eigvector[:,:l] #sort eigenvectors, only keep l
    reduced_wavelength_data= np.dot(eigvec.T,r.T) #np.dot(eigvec.T, np.dot(eigvec,r.T))
    if project == False:
        return reduced_wavelength_data.T # get the reduced wavelength weights
    else: 
        return np.dot(eigvec, reduced_wavelength_data).T # multiply eigenvectors by 
                                                        # weights to get approximate spectrum
                                                        
                                                        
plt.plot(logwave, PCA(5,flux)[1,:], label = 'l = 5')
plt.plot(logwave, flux[1,:], label = 'original data')

plt.ylabel('normalized 0-mean flux', fontsize = 16)
plt.xlabel('wavelength [$A$]', fontsize = 16)
plt.legend()

#1h)
coeff_list = PCA(5, flux, project = False)

plt.scatter(coeff_list[:,0], coeff_list[:,1])
plt.xlabel('c_0')
plt.ylabel('c_1')
plt.title('c_0 vs c_1')

plt.scatter(coeff_list[:,1], coeff_list[:,2])
plt.xlabel('c_1')
plt.ylabel('c_2')
plt.title('c_1 vs c_2')

#1g)

    
def PCA1(l, r, eigvector, project = True):
    """
    Perform PCA dimensionality reduction
    --------------------------------------------------------------------------------------
    """

    eigvec=eigvector[:,:l] #sort eigenvectors, only keep l
    reduced_wavelength_data= np.dot(eigvec.T,r.T) #np.dot(eigvec.T, np.dot(eigvec,r.T))
    if project == False:
        return reduced_wavelength_data.T # get the reduced wavelength weights
    else: 
        return np.dot(eigvec, reduced_wavelength_data).T # multiply eigenvectors by 
                                                        # weights to get approximate spectrum

ev = sorted_eigs(flux)
def fractional_error(l, r):
    approx = PCA1(l, flux, ev)
    residual  = r - approx
    fractional_res = residual**2
    return np.sum(fractional_res)


frac_error_list = []
for i in range(0, 20):
    error = fractional_error(i, flux)
    frac_error_list.append(error)

frac_error_list = np.array(frac_error_list)
plt.plot(range(0, 20), frac_error_list)
plt.xlabel('Number of PCA Vectors Used')
plt.ylabel('Squared Fractional Residuals')
plt.title('Squared Fractional Error vs ')
