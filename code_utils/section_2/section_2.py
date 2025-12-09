import numpy as np
from scipy.integrate import quad 
import math
from scipy.special import gamma

def Gamma(x):
    return gamma(x)
    

def kernel_comte_renault(x, g, alpha, k):
    def integrand(u, k, alpha):
        return math.exp(k*u)*u**alpha
    
    int_arg, _ = quad(integrand, 0, x, args=(k, alpha))
    r = (g/Gamma(1+alpha))*(x**alpha - k*math.exp(-k*x)*int_arg)
    return r 

def simulate_paths_basic_MC(M : int, n : int, T : float, y_0, ln_sigma_0, rho, k, g, alpha, r, q):
    '''
    Args: 
        M (int) : number of MC paths
        n (int) : number of timesteps, must be normalized to 1 ie, 1/n is the nr of time steps in 1 year
        T (int) : time to maturity.
        
        
    '''
    
    dt = 1/n
    
    nr_of_steps = int(round(n * T))
    
    sqrt_dt = np.sqrt(dt)
    
    W1 = np.random.normal(size=(M, nr_of_steps))
    W2 = np.random.normal(size=(M, nr_of_steps))
    
    W2 = rho*W1 + np.sqrt(1-rho**2)*W2
    
    dW1 = sqrt_dt*W1
    dW2 = sqrt_dt*W2
    
    
    
    a_vec = np.array([kernel_comte_renault(j*dt, g, alpha, k) for j in range(1, nr_of_steps+1)]) # kernell computation 
        
    
    X = np.empty((M, nr_of_steps))
    for i in range(nr_of_steps):
        X[:, i] = (dW2[:, :i+1] * a_vec[i::-1]).sum(axis=1) # volterra convolution 
    

       
        
    ln_sigma = ln_sigma_0 + np.concatenate([np.zeros((M, 1)), X], axis=1)
    
    sigma = np.exp(ln_sigma)
    
    
    incr = (r - q) * dt + sigma[:, :-1] * dW1 - 0.5 * (sigma[:, :-1]**2) * dt

    Y = np.empty((M, nr_of_steps + 1))
    Y[:, 0] = y_0
    Y[:, 1:] = y_0 + np.cumsum(incr, axis=1)

    
    S = np.exp(Y)

    
    Y_mean = Y.mean(axis=0)
    mc_error = 3 * S.std(axis=0, ddof=1) / np.sqrt(M)
    vol_mean = sigma.mean(axis=0)
    S_mean = S.mean(axis=0)
    
    
    
    return Y_mean, vol_mean, mc_error, S_mean

params_general_mc = {
    "j" : 10000, # number of paths in mc
    "n" : 252, # number of steps in 1 year
    "T" : 1.23, # year of simulation
    "r" : 0.03,
    "q" : 0
}

params_svm_c_r = {
    "g" : 0.01,
    "k" : 1,
    "alpha" : 0.2,
    "rho" : 0,
    "ln_sigma_0" : np.log(1)
}