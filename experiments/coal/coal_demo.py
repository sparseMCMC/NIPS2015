import numpy as np
import GPy
from mcmcGP import SGPMCMC, HMC, BinnedPoisson, AHMC
from jug import TaskGenerator as TG, barrier


def load_coal():
    X = np.loadtxt('coal.csv')
    return X



def build_kern():
    kern = GPy.kern.RBF(1, lengthscale=10.)+ GPy.kern.White(1)

    kern.rbf.lengthscale.set_prior(GPy.priors.Gamma(1.1, .1))
    kern.rbf.variance.set_prior(GPy.priors.Gamma(1., .3))
    kern.white.fix(1e-6)
    return kern


def build_vb(dataset, num_inducing):
    X = dataset['Xtrain']
    Z = np.linspace(X.min(), X.max(), num_inducing).reshape(-1, 1)

    lik = BinnedPoisson(np.diff(X.flat)[0])
    kern = build_kern()

    m = GPy.core.SVGP(X=X.reshape(-1, 1),
                      Y=dataset['Ytrain'].reshape(-1, 1),
                      Z=Z, kernel=kern, likelihood=lik)
    m.Z.fix()
    return m


def build_mc_sparse(dataset, num_inducing):
    X = dataset['Xtrain']
    Z = np.linspace(X.min(), X.max(), num_inducing).reshape(-1, 1)

    lik = BinnedPoisson(np.diff(X.flat)[0])

    kern = build_kern()

    m = SGPMCMC(X=X.reshape(-1, 1),
                Y=dataset['Ytrain'].reshape(-1, 1),
                Z=Z, kernel=kern, likelihood=lik)
    m.Z.fix()
    return m


@TG
def optimize_vb(dataset, num_inducing):
    m = build_vb(dataset, num_inducing)
    m.Z.fix()
    m.optimize('bfgs', max_iters=1, messages=1)
    m.Z.unfix()
    m.optimize('bfgs', max_iters=100, messages=1)
    return m[:]


@TG
def init_mc_model_from_vb(dataset, num_inducing, vb_param):
    # take the optimized vb model, and use it to init the mcmc mode
    m_mc = build_mc_sparse(dataset, num_inducing)
    m_vb = build_vb(dataset, num_inducing)
    m_vb[:] = vb_param

    m_mc.Z[:] = m_vb.Z[:]
    m_mc.Z.fix()
    m_mc.kern[:] = m_vb.kern[:]

    L = GPy.util.choleskies.flat_to_triang(m_vb.q_u_chol)[0, :, :]
    u_sample = np.dot(L, np.random.randn(m_vb.num_inducing))
    u_sample += m_vb.q_u_mean.flatten()
    L = GPy.util.linalg.jitchol(m_mc.kern.K(m_mc.Z))
    v_sample, _ = GPy.util.linalg.dtrtrs(L, u_sample)
    m_mc.V[:] = v_sample.reshape(-1, 1)
    return m_mc[:]


def split_train_test(X, seed, Ntrain, bins):
    np.random.seed(seed)
    i = np.random.permutation(X.size)
    Xtrain, Xtest = X[i[:Ntrain]], X[i[Ntrain:]]
    Ytrain = np.histogram(Xtrain, bins)[0]
    Ytest = np.histogram(Xtest, bins)[0]
    Xtrain = bins[:-1] + np.diff(bins)
    dataset = dict(Ytrain=Ytrain, Ytest=Ytest, Xtrain=Xtrain)

    return dataset


@TG
def run_hmc(dataset, x0, N, num_inducing):
    m = build_mc_sparse(dataset, num_inducing)
    m[:] = x0

    def f(x):
        return (-a for a in m._objective_grads(x))

    _, xopt, m_opt = AHMC(f, m.optimizer_array, 20, 40)

    samples, rate = HMC(f, N, Lmax=xopt[1], epsilon=np.exp(xopt[0]), x0=m.optimizer_array)
    return samples


@TG
def get_samples_vb(dataset, num_inducing, vb_param, num_samples):
    m = build_vb(dataset, num_inducing)
    m[:] = vb_param
    mu, var = m._raw_predict(m.X, full_cov=True)
    samples = np.random.multivariate_normal(mu.flat, var.squeeze(), num_samples).T
    return samples


@TG
def log_predictive_density(dataset, fsamples):
    lik = BinnedPoisson(binsize=np.diff(dataset['Xtrain'].flat)[0])
    Ytest = dataset['Ytest']
    pdfs = lik.pdf(fsamples, Ytest.reshape(-1, 1))
    pdfs = np.mean(pdfs, 1)  # average the density across samples
    logpdfs = np.log(pdfs)
    return np.mean(logpdfs)  # average the log density across bins


@TG
def get_samples_mc(dataset, mc_param, num_inducing, samples, samples_per_sample):
    m = build_mc_sparse(dataset, num_inducing)
    m[:]= mc_param
    ms, vs = [], []
    for s in samples:
        m.optimizer_array = s
        mui, vi = m.predict_raw(m.X)
        vi = np.clip(vi, 0, np.inf)
        ms.append(mui)
        vs.append(vi)

    print samples.shape
    fsamples = np.hstack([
        np.random.randn(mu.shape[0], samples_per_sample)*np.sqrt(var) + mu
        for mu, var in zip(ms, vs)])
    return fsamples


class Experiment:
    def __init__(self, seed, Ntrain,  num_grid, num_inducing, num_samples):
        self.seed, self.Ntrain,  self.num_grid, self.num_inducing, num_samples \
            = seed, Ntrain,  num_grid, num_inducing, num_samples
        np.random.seed(seed)

        X = load_coal()
        bins = np.linspace(X.min(), X.max(), num_grid+1)

        self.data = split_train_test(X, seed, Ntrain, bins)

        self.param_vb = optimize_vb(self.data, num_inducing)
        self.param_mc = init_mc_model_from_vb(self.data, num_inducing, self.param_vb)

        self.samples = run_hmc(self.data, self.param_mc, num_samples, num_inducing)

        self.fsamples_vb = get_samples_vb(self.data, num_inducing, self.param_vb, num_samples=1000)
        self.log_pred_vb = log_predictive_density(self.data, self.fsamples_vb)
        self.fsamples_mc = get_samples_mc(self.data, self.param_mc, num_inducing, self.samples[0::5], samples_per_sample=5)
        self.log_pred_mc = log_predictive_density(self.data, self.fsamples_mc)


num_samples = 20
Ms = [30]
grids = [100]
seeds = range(10)
experiments = [
    Experiment(seed, Ntrain=95, num_grid=ng,
               num_inducing=M, num_samples=num_samples)
    for seed in seeds for ng in grids for M in Ms]
