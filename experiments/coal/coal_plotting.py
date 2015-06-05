import numpy as np
import GPy
from mcmcGP import BinnedPoisson, GPMCMC, SGPMCMC, HMC
from jug import TaskGenerator as TG

def load_coal():
    X = np.loadtxt('coal.csv')
    return X

#@TG
def build_vb(bin_mids, counts, num_inducing, seed):
    np.random.seed(seed)

    # grid Z
    Z = np.linspace(bin_mids.min(), bin_mids.max(), num_inducing).reshape(-1,1)

    lik = BinnedPoisson(np.diff(bin_mids.flat)[0])
    kern = GPy.kern.RBF(1, lengthscale=10.)+GPy.kern.White(1, 1e-2) + GPy.kern.Bias(1)

    #pr = GPy.priors.Gamma(1., .3)
    #kern.set_prior(pr)
    return GPy.core.SVGP(X=bin_mids.reshape(-1,1), Y=counts.reshape(-1,1), Z=Z, kernel=kern, likelihood=lik)

#@TG
def build_mc_sparse(bin_mids, counts, num_inducing, seed):
    # grid Z
    Z = np.linspace(bin_mids.min(), bin_mids.max(), num_inducing).reshape(-1,1)

    lik = BinnedPoisson(np.diff(bin_mids.flat)[0])
    kern = GPy.kern.RBF(1, lengthscale=10.)+GPy.kern.White(1, 1e-2) + GPy.kern.Bias(1)
    kern.rbf.lengthscale.set_prior(GPy.priors.Gamma(1., .1))
    kern.rbf.variance.set_prior(GPy.priors.Gamma(1., .3))
    kern.bias.variance.set_prior(GPy.priors.Gamma(1., .3))
    kern.white.variance.fix(1e-6)

    return SGPMCMC(X=bin_mids.reshape(-1,1), Y=counts.reshape(-1,1), Z=Z, kernel=kern, likelihood=lik)

#@TG
def optimize_vb(m, max_iters=1000):
    m.kern.fix()
    m.optimize('bfgs', max_iters=100,messages=True)
    m.kern.constrain_positive()
    m.optimize('bfgs', max_iters=max_iters, messages=True)
    return m

#@TG
def init_mc_model_from_vb(m_mcmc, m_vb):
    #take the optimized vb model, and use it to init the mcmc mode
    m_mcmc.kern[:] = m_vb.kern[:]
    m_mcmc.Z[:] = m_vb.Z[:]
    m_mcmc.Z.fix()
    L = GPy.util.choleskies.flat_to_triang(m_vb.q_u_chol)[0,:,:]
    u_sample = np.dot(L, np.random.randn(m_vb.num_inducing))
    u_sample += m_vb.q_u_mean.flatten()
    L = GPy.util.linalg.jitchol(m_mcmc.kern.K(m_mcmc.Z))
    v_sample, _ = GPy.util.linalg.dtrtrs(L, u_sample)
    #v_sample = np.dot(L, np.random.randn(L.shape[0],1)) + m_vb.q_u_mean
    m_mcmc.V[:] = v_sample.reshape(-1,1)
    return m_mcmc

#@TG
def split_train_test(X, seed, Ntrain, bins):
    np.random.seed(seed)
    i = np.random.permutation(X.size)
    Xtrain, Xtest = X[i[:Ntrain]], X[i[Ntrain:]]
    Ytrain = np.histogram(Xtrain, bins)[0]
    Ytest = np.histogram(Xtest, bins)[0]
    dataset = dict(Ytrain=Ytrain, Ytest=Ytest)

    return dataset

#@TG
def run_hmc(m, N):
    def f(x):
        return (-a for a in m._objective_grads(x))
    samples, rate = HMC(f, N, Lmax=40, epsilon=0.08, x0=m.optimizer_array)
    return samples

#@TG
def get_samples_vb(m, num_samples):
    mu, var = m._raw_predict(m.X, full_cov=True)
    samples = np.random.multivariate_normal(mu.flat, var.squeeze(), num_samples).T
    return samples
#@TG
def log_predictive_density(m, fsamples, Ytest):
    pdfs = m.likelihood.pdf(fsamples, Ytest.reshape(-1,1))
    pdfs = np.mean(pdfs,1) # average the density across samples
    logpdfs = np.log(pdfs)
    return np.mean(logpdfs) # average the log density across bins

#@TG
def get_samples_mc(m, samples, numsamples):
    ms, vs = [],[]
    for s in samples:
        m.optimizer_array = s
        mui, vi = m.predict_raw(m.X, full_cov=True)
        ms.append(mui); vs.append(vi)

    samples = np.hstack([np.random.multivariate_normal(mu.flat, var, numsamples).T for mu, var in zip(ms, vs)])
    return samples






class Experiment:
    def __init__(self, seed, Ntrain,  num_grid, num_inducing, num_samples, vb_iterations):
        self.seed, self.Ntrain,  self.num_grid, self.num_inducing, num_samples = seed, Ntrain,  num_grid, num_inducing, num_samples
        np.random.seed(seed)

        X = load_coal()
        bins = np.linspace(X.min(), X.max(), num_grid+1)
        bin_mids = bins[:-1]+np.diff(bins)

        self.data = split_train_test(X, seed, Ntrain, bins)

        self.m_vb = build_vb(bin_mids, self.data['Ytrain'], num_inducing, seed)
        self.m_vb = optimize_vb(self.m_vb,vb_iterations)

        self.m_mc = build_mc_sparse(bin_mids, self.data['Ytrain'], num_inducing, seed)
        self.m_mc = init_mc_model_from_vb(self.m_mc, self.m_vb)

	self.m_mc.kern.rbf.constrain(GPy.core.parameterization.transformations.Exponent())
	self.m_mc.kern.bias.constrain(GPy.core.parameterization.transformations.Exponent())
        self.samples = run_hmc(self.m_mc, num_samples)

        self.fsamples_vb = get_samples_vb(self.m_vb, 10000)
        self.log_pred_vb = log_predictive_density(self.m_vb, self.fsamples_vb, self.data['Ytest'])
        self.fsamples_mc = get_samples_mc(self.m_mc, self.samples[::2], 10)
        self.log_pred_mc = log_predictive_density(self.m_mc, self.fsamples_mc, self.data['Ytest'])

num_samples = 10000
vb_iterations = 10000
M = 20
grid = 50
seed = 0
experiment = Experiment(seed, Ntrain=96, num_grid=grid, num_inducing=M, num_samples=num_samples, vb_iterations=vb_iterations)

from matplotlib import pylab as plt
from matplotlib2tikz import save as save_tikz
plt.figure()
X = experiment.m_vb.X
bw = experiment.m_vb.likelihood.binsize
plt.plot(X,np.mean(np.exp(experiment.fsamples_vb)*bw,axis=1) ,'b' ,label='VB+Gaussian')
plt.plot(X,np.mean(np.exp(experiment.fsamples_mc)*bw,axis=1) ,'r' ,label='VB+MCMC')
plt.legend()
plt.plot(X,np.percentile(np.exp(experiment.fsamples_vb)*bw, 5,axis=1), 'b-')
plt.plot(X,np.percentile(np.exp(experiment.fsamples_vb)*bw,95,axis=1), 'b-')
plt.plot(X,np.percentile(np.exp(experiment.fsamples_mc)*bw, 5,axis=1), 'r-')
plt.plot(X,np.percentile(np.exp(experiment.fsamples_mc)*bw,95,axis=1), 'r-')
#load and plot raw data
X = load_coal()
plt.plot(X, X*0, 'k|')
plt.xlabel('time (years)')
plt.ylabel('rate')
plt.ylim(-.05, 1)
plt.xlim(X.min(), X.max())
save_tikz('coal_rates.tikz',figurewidth='\\figurewidth', figureheight = '\\figureheight')

plt.figure()
#trans = GPy.core.parameterization.transformations.Logexp()
trans = GPy.core.parameterization.transformations.Exponent()
plt.hist(trans.f(experiment.samples[:,0]), 100, normed=True)
plt.xlabel('signal varaince')
#save_tikz('coal_variance.tikz',figurewidth='\\figurewidth', figureheight = '\\figureheight')
np.savetxt('coal_var_samples',trans.f(experiment.samples[:,0]))
plt.figure()
plt.hist(trans.f(experiment.samples[:,1]), 100, normed=True)
plt.xlabel('lengthscale')
#save_tikz('coal_lengthscale.tikz',figurewidth='\\figurewidth', figureheight = '\\figureheight')
np.savetxt('coal_ls_samples',trans.f(experiment.samples[:,1]))

#plota scatter of variance, ls
variances = trans.f(experiment.samples[:,0])
lengthscales = trans.f(experiment.samples[:,1])
plt.figure()
plt.plot(lengthscales, variances, 'k.')
save_tikz('coal_theta.tikz',figurewidth='\\figurewidth', figureheight = '\\figureheight')



