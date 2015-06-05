from __future__ import division, print_function
import numpy as np
from numpy import log, exp, sqrt
import GPy

from ahmc import HMC

"""
AA implements the Ancillary Augmentation scheme to sample hyper-parameters and latent variables in GP models

AA uses a Gaussian proposal with variance alpha for the hyper-parameters and HMC for the latent variables

"""

def AA(f, num_samples, Lmax, epsilon, alpha, x0, m_mcmc, kernel_priors, verbose=False):
    nvars = m_mcmc.optimizer_array.size
    nlatent = m_mcmc.V.size
    ntheta = nvars - nlatent

    samples = np.zeros((num_samples, nvars))
    samples[0,:] = x0

    accept_count_batch = np.zeros(2)
    accept_count = np.zeros(2)

    curr_theta = m_mcmc.optimizer_array[0:ntheta] * 1.0    
    curr_v = m_mcmc.optimizer_array[ntheta:nvars] * 1.0

    for t in range(num_samples-1):

        ## Output acceptance rate every 100 iterations
        if(((t+1) % 100) == 0):
            print("Iteration: ", t+1, "\t Acc Rate: ", 1. * accept_count_batch, "%")

            if(t < 1000):
                if (accept_count_batch[0] < 10.): alpha = alpha * 0.8
                if (accept_count_batch[1] < 50.): epsilon = epsilon * 0.8

                if (accept_count_batch[0] > 40.): alpha = alpha * 1.2
                if (accept_count_batch[1] > 90.): epsilon = epsilon * 1.2

            accept_count_batch = np.zeros(2)

        # MH on the hyper-parameters - begin leapfrogging
        curr_logp = m_mcmc._objective(np.concatenate((curr_theta, curr_v))) * -1

        prop_theta = curr_theta + alpha * np.random.randn(ntheta)
        prop_logp = m_mcmc._objective(np.concatenate((prop_theta, curr_v))) * -1
        
        log_accept_ratio = prop_logp - curr_logp
        logu = np.log(np.random.rand())

        if logu < log_accept_ratio:
            samples[t+1,0:ntheta] = prop_theta[0:ntheta]
            curr_theta = np.copy(prop_theta)
            accept_count_batch[0] += 1
            accept_count[0] += 1
        else:
            samples[t+1,0:ntheta] = samples[t,0:ntheta]

        m_mcmc.kern.fix(warning=False)
        m_mcmc.kern.unset_priors()

        curr_logp, curr_grad = f(curr_v)

        pt = np.random.randn(nlatent)
        Lt = np.random.randint(1, Lmax+1)
        
        # HMC on the latent variables - begin leapfrogging
        premature_reject = False
        x = np.copy(curr_v)
        p = pt + 0.5 * epsilon * curr_grad
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
            samples[t+1,ntheta:nvars] = samples[t,ntheta:nvars]
            m_mcmc.kern.unfix()
            m_mcmc.kern = kernel_priors(m_mcmc.kern)
            continue

        log_accept_ratio = logprob - 0.5*p.dot(p) - curr_logp + 0.5*pt.dot(pt)
        logu = np.log(np.random.rand())

        if logu < log_accept_ratio:
            samples[t+1,ntheta:nvars] = x
            curr_v = np.copy(x)
            accept_count_batch[1] += 1
            accept_count[1] += 1
        else:
            samples[t+1,ntheta:nvars] = samples[t,ntheta:nvars]

        m_mcmc.kern.unfix()
        m_mcmc.kern = kernel_priors(m_mcmc.kern)
        
    return samples, (accept_count*1.0)/num_samples


def ESJD(samples):
    diffs = np.diff(samples, axis=0)
    return np.mean(np.sum(np.square(diffs),axis=1))

def ACCRATE_AA(samples):
    nvars = samples.shape[1]
    ns = samples.shape[0]
    acc_theta = np.sum(np.diff(samples[:,0], axis=0) != 0) * 1.0 / ns
    acc_latent = np.sum(np.diff(samples[:,nvars-1], axis=0) != 0) * 1.0 / ns
    return ((acc_theta-0.3)**2 + (acc_latent-0.8)**2)

def median_SJD(samples):
    diffs = np.diff(samples, axis=0)
    return np.median(np.sum(np.square(diffs),axis=1))


def AAA(f, m_mcmc, kernel_priors, theta0, num_steps, samples_per_step, epsilon_bounds=[1e-2, 1e-1], L_bounds=[5,50], alpha_bounds=[1e-2, 1e-1], grid_res=8, verbose=False, criterion = "ESJD"):
    """
    Use bayesian optimization to adjust the parameters of HMC using expected jump size criterion
    """

    nvars = m_mcmc.optimizer_array.size
    nlatent = m_mcmc.V.size
    ntheta = nvars - nlatent

    #set up the kernel
    xx_eps, xx_L, xx_alpha = np.meshgrid(np.linspace(np.log(epsilon_bounds[0]),np.log(epsilon_bounds[1]),grid_res), np.arange(L_bounds[0], L_bounds[1], 3), np.linspace(np.log(alpha_bounds[0]),np.log(alpha_bounds[1]),grid_res))
    X_all = np.vstack((xx_eps.flatten(), xx_L.flatten(), xx_alpha.flatten())).T

    X_tried = []
    samples =[]
    Y_tried = []

    if(criterion == "ESJD"):
        objective_function = ESJD

    if(criterion == "median_SJD"):
        objective_function = median_SJD

    if(criterion == "ACCRATE_AA"):
        objective_function = ACCRATE_AA


    #set up first point
    X = np.array([np.mean(np.log(epsilon_bounds)), int(np.mean(L_bounds)), np.mean(np.log(alpha_bounds))])

    for i in range(num_steps):
        #run AA scheme
        X_tried.append(X)
        samples.append(AA(f, samples_per_step, Lmax=X[1], epsilon=np.exp(X[0]), alpha=np.exp(X[2]), x0=theta0, m_mcmc=m_mcmc, kernel_priors=kernel_priors, verbose=verbose)[0])
        # Y_tried.append(np.log(1+objective_function(samples[-1][:,0:ntheta]) / ntheta + objective_function(samples[-1][:,ntheta:nvars])/(np.sqrt(X[1]) * nlatent) ))
        Y_tried.append(np.log(1+objective_function(samples[-1])))

        theta0 = samples[-1][-1]

        #build a GP model of the objective fn
        kern = GPy.kern.RBF(3, ARD=True)
        kern.lengthscale = 0.3*np.array([np.diff(np.log(epsilon_bounds)), np.diff(L_bounds), np.diff(np.log(alpha_bounds))]).flatten()
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
    kern = GPy.kern.RBF(3, ARD=True)
    kern.lengthscale = 0.3*np.array([np.diff(np.log(epsilon_bounds)), np.diff(L_bounds), np.diff(np.log(alpha_bounds))]).flatten()
    kern.fix(warning=False)
    Y = np.array(Y_tried)
    Y=(Y-np.mean(Y))/np.std(Y)
    model = GPy.models.GPRegression(np.vstack(X_tried), Y.reshape(-1,1), kernel=kern)
    model.optimize()
    utility, _ = model._raw_predict(X_all)
    index = np.argmax(utility.flatten())
    return samples, X_all[index], model
    


def HMC_safe(f, num_samples, Lmax, epsilon, x0, verbose=False, option_hmc = 0, mean_approx = None, cov_approx = None):
    D = x0.size
    samples = np.zeros((num_samples, D))
    samples[0] = x0
    logprob, grad = f(x0)


    if(option_hmc != 0):
        raise NotImplementedError

    accept_count_batch = 0
    accept_count = 0

    for t in range(num_samples-1):

        ## Output acceptance rate every 100 iterations
        if(((t+1) % 100) == 0):
            print("Iteration: ", t+1, "\t Acc Rate: ", 1. * accept_count_batch, "%")

            if(t < 1000):
                if (accept_count_batch < 50.): epsilon = epsilon * 0.9
                if (accept_count_batch > 90.): epsilon = epsilon * 1.1
                
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


def AHMC_lessL(f, theta0, num_steps, samples_per_step, epsilon_bounds=[1e-2, 1e-1], L_bounds=[5,50], grid_res=20, verbose=False, option_hmc = 0, mean_approx = None, cov_approx = None, criterion = "ESJD"):
    """
    Use bayesian optimization to adjust the parameters of HMC using expected jump size criterion
    """
    #set up the kernel
    xx_eps, xx_L = np.meshgrid(np.linspace(np.log(epsilon_bounds[0]),np.log(epsilon_bounds[1]),grid_res), np.arange(L_bounds[0], L_bounds[1], 3))
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
