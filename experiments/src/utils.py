import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln, logsumexp
from numpy import linalg as la
from matplotlib.patches import Ellipse



def log_wishart_cst(invV, nu):
    """computes the constant terms in the expression of the log of the Wishart distribution 
    nu : int
        degrees of freedom
    invV : 2D numpy array 
        inverse of the scale matrix
    """
    D = len(invV)   # dimension of samples
    return + 0.5 * nu * np.log(np.linalg.det(invV))   - 0.5 * nu * D * np.log(2)   - 0.25 * D * (D-1) * np.log(np.pi) - gammaln(0.5 * (nu - np.arange(D))).sum()



def plot_ellipse(mu, cov, ax, **kwargs):
    """Displays an ellipse of a multivariate normal distribution
    
    Parameters:
        mu : 1D numpy array of shape (D) 
            mean of the distribution
        cov 2D numpy array of shape (D x D) 
            covariance matrix of the distribution
        ax : plt.Axes 
            axes on which to display the ellipse
        kwargs :
            other arguments for the class Ellipse
    """
    c = 4  # scaling coeff.
    Lambda, Q = la.eig(cov)  # eigenvalues and eigenvectors 
    
    # width and height of the ellipse
    width, heigth = 2 * np.sqrt(c * Lambda)

    # compute the value of the angle theta (in degree)
    if cov[1,0]:
        theta = 180 * np.arctan(Q[1,0] / Q[0,0]) / np.pi
    else: 
        theta = 0
        
    # create the ellipse
    if 'fc' in kwargs.keys():    
        kwargs['fc'] = matplotlib.colors.to_rgba(kwargs['fc'], 0.05)     # facecolor with low alpha value
    else:
        kwargs['fc'] = 'None'
    ellipse = Ellipse(mu, width, heigth, angle=theta, **kwargs)
    
    return ax.add_patch(ellipse)


    
