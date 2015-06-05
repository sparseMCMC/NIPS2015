from __future__ import division, print_function
import numpy as np
from numpy import log, exp, sqrt
import GPy

"""
HMC implements three variants of Hybrid Monte Carlo

The input variable option_hmc determines the kind of HMC sampler that will be run - in particular

- option_hmc = 0 (DEFAULT) implements the standard HMC

"""

def HMC(f, num_samples, Lmax, epsilon, x0, verbose=False, option_hmc = 0, mean_approx = None, cov_approx = None):
    D = x0.size
    samples = np.zeros((num_samples, D))
    samples[0] = x0
    logprob, grad = f(x0)

    ## Precompute the rotation matrix if using split HMC or the guiding Hamiltonian
    ## In these cases the potential energy is a quadratic form involving the inverse of the covariance of the approximating Gaussian
    if(option_hmc != 0):
        ## Compute eigenvalues and eigenvectors of the covariance of the Gaussian approximation
        ## These are useful to rotate the coordinates of the harmonic oscillator in D dimensions making the D dynamics independent
        eigen_cov_approx = np.linalg.eig(cov_approx)
        eigenvect_cov_approx = eigen_cov_approx[1]
        eigenval_cov_approx = eigen_cov_approx[0]

        ## omega is the vector of the D angular frequencies sqrt(1/lambda_i) where lambda_i is the ith eigenvalue
        omega = np.zeros(D)
        for i in range(0,D):
            omega[i] = np.sqrt(1/(eigenval_cov_approx[i]))

    accept_count_batch = 0
    accept_count = 0

    for t in range(num_samples-1):

        ## Output acceptance rate every 100 iterations
        if(((t+1) % 100) == 0):
            print("Iteration: ", t+1, "\t Acc Rate: ", 1. * accept_count_batch, "%")
            accept_count_batch = 0

        logprob_t, grad_t = logprob, grad.copy()
        pt = np.random.randn(D)
        Lt = np.random.randint(1, Lmax+1)


        # Standard HMC - begin leapfrogging
        premature_reject = False
        if(option_hmc == 0):
            x = samples[t].copy()
            p = pt + 0.5 * epsilon * grad
            for l in range(Lt):
                x  += epsilon * p
                logprob, grad = f(x)
                if np.any(np.isnan(grad)):
                    premature_reject = True
                    break
                p += epsilon * grad
            p -= 0.5*epsilon * grad
        #leapfrogging done
        if premature_reject:
            print("warning: numerical instability. rejecting this proposal prematurely")
            samples[t+1] = samples[t]
            logprob, grad = logprob_t, grad_t
            continue

        if(option_hmc == 1):
            raise NotImplementedError
        
        if(option_hmc == 2):
            raise NotImplementedError

        log_accept_ratio = logprob - 0.5*p.dot(p) - logprob_t + 0.5*pt.dot(pt)
        logu = np.log(np.random.rand())

        if verbose:
            print('sample number {:<4} steps: {:<3}, eps: {:.2f} logprob: {:.2f} accept_prob: {:.2f}, {} (accepted {:.2%})'.format(t,Lt, epsilon, logprob, np.fmin(1, np.exp(log_accept_ratio)), 'rejecting' if logu>log_accept_ratio else 'accepting', accept_count/(t+1)))
        if logu < log_accept_ratio:
            samples[t+1] = x
            accept_count_batch += 1
            accept_count += 1
        else:
            samples[t+1] = samples[t]
            logprob, grad = logprob_t, grad_t
    return samples, (accept_count*1.0)/num_samples


def ESJD(samples):
    diffs = np.diff(samples, axis=0)
    return np.mean(np.sum(np.square(diffs),axis=1))

def median_SJD(samples):
    diffs = np.diff(samples, axis=0)
    return np.median(np.sum(np.square(diffs),axis=1))


def AHMC(f, theta0, num_steps, samples_per_step, epsilon_bounds=[1e-2, 1e-1], L_bounds=[5,50], grid_res=20, verbose=False, option_hmc = 0, mean_approx = None, cov_approx = None, criterion = "ESJD"):
    """
    Use bayesian optimization to adjust the parameters of HMC using expected jump size criterion
    """
    #set up the kernel
    xx_eps, xx_L = np.meshgrid(np.linspace(np.log(epsilon_bounds[0]),np.log(epsilon_bounds[1]),grid_res), np.arange(L_bounds[0], L_bounds[1]))
    X_all = np.vstack((xx_eps.flatten(), xx_L.flatten())).T

    X_tried = []
    samples =[]
    Y_tried = []

    if(criterion == "ESJD"):
        objective_function = ESJD

    if(criterion == "median_SJD"):
        objective_function = median_SJD

    #set up first point
    X = np.array([np.mean(np.log(epsilon_bounds)), int(np.mean(L_bounds))])

    for i in range(num_steps):
        #run HMC
        X_tried.append(X)
        samples.append(HMC(f, samples_per_step, Lmax=X[1], epsilon=np.exp(X[0]), x0=theta0, verbose=verbose, option_hmc=option_hmc, mean_approx=mean_approx, cov_approx=cov_approx)[0])
        Y_tried.append(np.log(1+objective_function(samples[-1])/np.sqrt(X[1])))

        theta0 = samples[-1][-1]

        #build a GP model of the objective fn
        kern = GPy.kern.RBF(2, ARD=True)
        kern.lengthscale = 0.3*np.array([np.diff(np.log(epsilon_bounds)), np.diff(L_bounds)]).flatten()
        kern.lengthscale.fix(warning=False)
        Y = np.array(Y_tried)
        if Y.size > 3:
            Y=(Y-np.mean(Y))/np.std(Y)
        model = GPy.models.GPRegression(np.vstack(X_tried), Y.reshape(-1,1), kernel=kern)
        model.optimize()
        #maximize the utility.
        means, v = model._raw_predict(X_all)
        std = np.sqrt(v)
        u = (-Y.max() + means)/std
        Phi, phi = GPy.util.univariate_Gaussian.std_norm_cdf(u), GPy.util.univariate_Gaussian.std_norm_pdf(u)
        utility = u*std*Phi + std*phi

        index = np.argmax(utility.flatten())
        X = X_all[index]

    #find the best point
    kern = GPy.kern.RBF(2, ARD=True)
    kern.lengthscale = 0.3*np.array([np.diff(np.log(epsilon_bounds)), np.diff(L_bounds)]).flatten()
    kern.fix(warning=False)
    Y = np.array(Y_tried)
    Y=(Y-np.mean(Y))/np.std(Y)
    model = GPy.models.GPRegression(np.vstack(X_tried), Y.reshape(-1,1), kernel=kern)
    model.optimize()
    utility, _ = model._raw_predict(X_all)
    index = np.argmax(utility.flatten())
    return samples, X_all[index], model
