import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.special import digamma, gammaln, logsumexp    # deriv. of the log(gamma), log |gamma|, log( sum( exp(.) ) )
from numpy import linalg as la
from matplotlib.patches import Ellipse

# importing useful functions 
from .utils import log_wishart_cst, plot_ellipse



class VariationalGaussianMixture():
    """class for mixture models"""

    def __init__(self, K, seed=2208, max_iter=200, beta0=None, nu0=None, invV0=None, display=False, plot_period=None):
        self.K = K                                          # initial nb of components
        self.rd = np.random.RandomState(seed)               # rd generator
        self.max_iter = max_iter                            # max nb of iterations 
        self.plot_period = plot_period or max_iter // 10    # period of the plotting
        self.display = display                              # to display or not the figure

        self.beta0 = beta0          # initial value of beta 
        self.nu0 = nu0              # initial degrees of freedom
        self.invV0 = invV0          # inverse of the initial scale matrix 



    # ------------ initialize_parameters ------------ 
    def initialize_parameters(self, X):
        """initializes the parameters of the model : 
            - nu0 : deg. of freedom of the Wishart dist.
            - inV0 : inverse of the Wishart dist.'s scale matrix
            - beta0 : cst beta used in the dist. of mu
            - mixing_coeffs : mixing coefficients (pi)
            - exp_sin : <sin> ; initialized using K-means
            - exp_T : <T> 
            - exp_log_det_T : <log|T|>
            - exp_mu : <mu>
            - exp_mu_muT : <mu.muT>

        Parameters:
        -----------
        X : 2D numpy array
            dataset examples
        """
        N, D = X.shape    # nb of samples, dim of the samples

        #  initializing <sin>
        exp_sin = np.zeros((N, self.K))
        label = KMeans(n_clusters=self.K, n_init=1).fit(X).labels_   # assigned labels to the samples
        exp_sin[np.arange(N), label] = 1
        
        # nu0, invV0, beta0, mixing_coeffs
        self.nu0 = self.nu0 or D
        self.invV0 = self.invV0 or np.atleast_2d(np.cov(X.T))
        self.beta0 = self.beta0 or 1
        self.mixing_coeffs = np.ones(self.K) / self.K

        # computing initial expectations : <T>, <log|T|>, <mu>, <mu.muT>
        exp_T =np.array([self.nu0 * np.linalg.inv(self.invV0) for _ in range(self.K)])   # <T>
        exp_log_det_T = np.zeros(self.K)        # <log|T|>
        exp_mu = np.zeros((self.K, D))          # <mu>
        exp_mu_muT = np.zeros((self.K, D, D))   # <mu.muT>
        
        # updating parameters (using computed expectations)
        self.update_parameters(X, exp_sin, exp_T, exp_log_det_T, exp_mu, exp_mu_muT)

    

    # ------------ update_parameters ------------     
    def update_parameters(self, X, exp_sin, exp_T, exp_log_det_T, exp_mu, exp_mu_muT):

        """ updates the parameters using computed expectations
        Parameters:
        -----------
        X : 2D numpy array (N x D)
            dataset examples
        exp_sin : 2D numpy array (N x K)
            <sin> 
        exp_T : 3D numpy array (K x D x D)
            <T>
        exp_log_det_T : 1D numpy array (K)
            <log|T|>
        exp_mu : 2D numpy array (K x D)
             <mu>
        exp_mu_muT : 3D numpy array (K x D x D)
            <mu.muT>
        """
        N, D = X.shape    # nb of samples, dim. of samples

        # updating nu 
        sum_exp_sin = exp_sin.sum(axis=0)     # nb of examples in each component 
        self.nu = self.nu0 + sum_exp_sin      # nu_t(i)

         # updating T 
        self.T = self.beta0 * np.identity(D) + exp_T * sum_exp_sin[:,np.newaxis,np.newaxis]  # T_mu(i); np.newaxis is used to adapt the shapes
        invT = np.linalg.inv(self.T)
        
        # updating invV and m
        self.invV = np.zeros((self.K, D, D))  
        self.m = np.zeros((self.K, D)) 
        for k in range(self.K):
            self.m[k] = invT[k].dot(exp_T[k].dot(exp_sin[:, k].dot(X))) 

            second_term = np.zeros((D, D))     #  2nd term of invV_k
            for n in range(N):
                second_term += exp_sin[n, k] * (np.outer(X[n], X[n]) - np.outer(X[n], exp_mu[k])   - np.outer(exp_mu[k], X[n]) +  exp_mu_muT[k])  
            self.invV[k] = self.invV0 + second_term  



    # ------------ compute_expectations ------------ 
    def compute_expectations(self, D):
        """computing expectation of the model's parameters

        Parameters
        ----------
        D : int
            dimension of samples

        Returns
        -------
        tuple
            expectations : <T>, <log|T|>, <mu>, <mu.muT>, <muT.mu>
        """
        exp_T = self.nu[:, np.newaxis, np.newaxis] * np.linalg.inv(self.invV)  # <T> 
        exp_log_det_T = np.sum(digamma(0.5 * (self.nu - np.arange(D)[:,np.newaxis])), axis=0)  + D * np.log(2) - np.log(np.linalg.det(self.invV))  # <log|T|>
        exp_mu = np.copy(self.m)   # <mu>

        # expectations <mu.muT>, <muT.mu>
        exp_mu_muT = np.zeros_like(self.T)
        exp_muT_mu = np.zeros(self.K)
        invT = np.linalg.inv(self.T)
        for k in range(self.K):
            exp_mu_muT[k] = invT[k] + np.outer(self.m[k], self.m[k]) 
            exp_muT_mu[k] = np.trace(invT[k]) + self.m[k].dot(self.m[k])

        return exp_T, exp_log_det_T, exp_mu, exp_mu_muT, exp_muT_mu 



    # ------------ compute_exp_sin ------------ 
    def compute_exp_sin(self, X, exp_T, exp_log_det_T, exp_mu, exp_mu_muT, exp_muT_mu):
        """ computes the expectation of s_in 

        Parameters:
        -----------
        X : 2D numpy array (N x D)
            dataset examples
        exp_T : 3D numpy array (K x D x D)
            <T>
        exp_log_det_T : 1D numpy array (K)
            <log|T|>
        exp_mu : 2D numpy array (K x D)
             <mu>
        exp_mu_muT : 3D numpy array (K x D x D)
            <mu.muT>
        exp_muT_mu : 1D numpy array (K)
            <muT.mu>

        Returns: 
        --------
        log_exp_sin : 2D numpy array (N x K)
            log <sin>
        log_p_hat : 2D numpyo array  (N x K)
            log <p_tilde> - sum(pi)
        """
        N, D = X.shape    # nb of samples, dim. of samples
        log_p_hat = np.zeros((N, self.K))    # = log(p_tilde) - sum(pi)
        for n in range(N):
            for k in range(self.K):
                log_p_hat[n, k] = 0.5 * exp_log_det_T[k] - 0.5 * np.trace(
                    exp_T[k].dot(np.outer(X[n], X[n]) - np.outer(X[n], exp_mu[k]) - np.outer(exp_mu[k], X[n]) + exp_mu_muT[k])) 
        
        log_p_tilde = log_p_hat + np.log(self.mixing_coeffs)  # log(p_tilde)
        log_exp_sin = log_p_tilde - logsumexp(log_p_tilde, axis=1)[:, np.newaxis]  
        return log_exp_sin, log_p_hat


    # ------------ compute_lower_bound ------------ 
    def compute_lower_bound(self, X, log_exp_sin, exp_sin, log_p_hat, exp_T, exp_log_det_T, exp_mu, exp_mu_muT, exp_muT_mu):
        """computes the lower bound of the marginal log-likelihood

        Parameters
        ----------
        X : 2D numpy array
            dataset examples
        log_exp_sin : 2D numpy array (N x K)
            log <sin>
        exp_sin : 2D numpy array (N x K)
            <sin>
        log_p_hat : 2D numpy array (N x K)
            log(p_tilde) - sum(pi)
        exp_T : 3D numpy array (K x D x D)
             <T>
        exp_log_det_T :  1D numpy array (K)
             <log|T|>
        exp_mu : 2D numpy array (K x D)
            <mu>
        exp_mu_muT : 3D numpy array (K x D x D)
            <mu.muT>
        exp_muT_mu : 1D numpy array (K)
            <muT.mu>

        Returns
        -------
        float
            lower bound of the marginal log-likelihood
        """
        N, D = X.shape

        ln_p_x = np.sum(exp_sin * (log_p_hat))  
        ln_p_z = np.sum(exp_sin * np.log(self.mixing_coeffs))  
        ln_p_mu = self.K * D * np.log(0.5 * self.beta0 / np.pi) - 0.5 * self.beta0 * np.sum(exp_muT_mu) 
        ln_p_T = self.K * log_wishart_cst(self.invV0, self.nu0) + 0.5 * (self.nu0 - D - 1) * exp_log_det_T.sum()  - 0.5 * np.trace(self.invV0 * exp_T.sum())  
        
        ln_q_z = np.sum(exp_sin * np.log(exp_sin))  
        ln_q_mu = - 0.5 * self.K * D * (1 + np.log(2 * np.pi))  + 0.5 * np.sum(np.log(np.linalg.det(self.T)))  
        ln_q_T  = np.sum([log_wishart_cst(self.invV[k], self.nu[k]) for k in range(self.K)])   + np.sum(0.5 * (self.nu - D - 1) * exp_log_det_T)  - np.sum(0.5 * np.trace(self.invV.dot(exp_T), axis1=1, axis2=2))  
        return ln_p_x + ln_p_z + ln_p_mu + ln_p_T - ln_q_z - ln_q_mu - ln_q_T


    # ------------ display_figure ------------ 
    def display_figure(self, X):
        """display the ellipe of the multivariate Guassian distributions

        Parameters
        ----------
        X : 2D numpy array
            dataset examples
        """
        assert X.shape[1] == 2, "Can only plot in 2D!"
        
        plt.figure(figsize=(6,6))
        
        # Scatter plot of the dataset
        plt.plot(*X.T, '.', c='k', alpha=0.6)

        # Display each component of the Gaussian mixture
        ax = plt.gca()
        for k in range(self.K):
            # plotting only components with significant mixing_coeffs values
            if self.mixing_coeffs[k] >= 1e-5:
                plot_ellipse(self.m[k], self.covs[k], ax=ax,
                    edgecolor='red', linestyle='--')



    # ------------ fit ------------ 
    def fit(self, X):
        """fits a mixture model to provided data (X)

        Parameters
        ----------
        X : 2D numpy array
            dataset of examples

        Returns
        -------
        VariationalGaussianMixture:     
            the fitted model
        """
        N, D = X.shape
        
        # initializing parameters of the model
        self.initialize_parameters(X)

        # lower bound of the marginal log-likelihood L(Q)
        self.lower_bound = np.zeros(self.max_iter)

        # optimization loop
        for i in range(self.max_iter):
            # E-step
            exp_T, exp_log_det_T, exp_mu, exp_mu_muT, exp_muT_mu  = self.compute_expectations(D)
            log_exp_sin, log_p_hat = self.compute_exp_sin(X, exp_T, exp_log_det_T, exp_mu, exp_mu_muT, exp_muT_mu )
            exp_sin = np.exp(log_exp_sin)
            # updating parameters (from computed expectations)
            self.update_parameters(X, exp_sin, exp_T, exp_log_det_T, exp_mu, exp_mu_muT)

            # M-step
            self.mixing_coeffs = exp_sin.sum(axis=0) / exp_sin.sum()

            # current lower bound of the marginal log-likelihood L(Q)
            self.lower_bound[i] = self.compute_lower_bound(X, log_exp_sin, exp_sin, log_p_hat, exp_T, exp_log_det_T, exp_mu, exp_mu_muT, exp_muT_mu)

            # display of current figure
            if self.display and D == 2 and i % self.plot_period == 0:
                self.covs = self.invV / self.nu[:, np.newaxis, np.newaxis]
                self.display_figure(X)
                plt.title(f'iteration {i}')
                plt.show()
        
        # display of the final figure 
        self.covs = self.invV / self.nu[:, np.newaxis, np.newaxis]
        if self.display and D == 2:
            self.display_figure(X)
            plt.title(f'iteration {i}')
            plt.show()

        return self
 