import numpy as np
import GPy
from mcmcGP import SGPMCMC, HMC, AHMC
from jug import TaskGenerator as TG
from scipy.cluster.vq import kmeans as scipy_kmeans
from scipy import io


def load_data(seed, ntrain, datasetName, num_inducing):
    d = io.loadmat('benchmarks.mat')[datasetName][0, 0]
    x, y = d[0], d[1]
    y = np.where(y == 1, 1, 0)  # data is stored as +-1, we use 1, 0

    # split into train, test sets
    np.random.seed(seed)
    index = np.random.permutation(x.shape[0])
    itrain, itest = index[:ntrain], index[ntrain:]
    xtrain, xtest = x[itrain], x[itest]
    ytrain, ytest = y[itrain], y[itest]

    # normalize using training data mean, std
    xmean, xstd = xtrain.mean(0), xtrain.std(0)
    xstd = np.where(xstd > 1e-6, xstd, 1.)
    xtrain, xtest = (xtrain-xmean)/xstd, (xtest-xmean)/xstd
    Z, _ = scipy_kmeans(xtrain, num_inducing)
    return dict(Xtrain=xtrain, Ytrain=ytrain, Xtest=xtest, Ytest=ytest, Z=Z)


def build_vb(dataset):
    X = dataset['Xtrain']
    Z = dataset['Z']

    lik = GPy.likelihoods.Bernoulli()
    kern = GPy.kern.RBF(X.shape[1], lengthscale=1.) + GPy.kern.White(1)

    kern.rbf.lengthscale.set_prior(GPy.priors.Gamma(1.1, 1))
    kern.rbf.variance.set_prior(GPy.priors.Gamma(1.1, 1))
    kern.white.fix(1e-6)

    m = GPy.core.SVGP(X=X, Y=dataset['Ytrain'],
                      Z=Z, kernel=kern, likelihood=lik)
    return m


def build_mc_sparse(dataset):
    X = dataset['Xtrain']
    Z = dataset['Z']

    lik = GPy.likelihoods.Bernoulli()
    kern = GPy.kern.RBF(X.shape[1], lengthscale=1.) + GPy.kern.White(1)

    kern.rbf.lengthscale.set_prior(GPy.priors.Gamma(1.1, 1))
    kern.rbf.variance.set_prior(GPy.priors.Gamma(1.1, 1))
    kern.white.fix(1e-6)

    m = SGPMCMC(X=X, Y=dataset['Ytrain'], Z=Z, kernel=kern, likelihood=lik)
    m.Z.fix()
    return m


@TG
def optimize_vb(dataset, fixZ):
    m = build_vb(dataset)
    m.Z.fix()
    m.optimize('bfgs', max_iters=100, messages=1)
    if not fixZ:
        m.Z.unfix()
    m.optimize('bfgs', max_iters=10000, messages=1)
    return m[:]


@TG
def init_mc_from_vb(dataset, vb_param):
    # take the optimized vb model, and use it to init the mcmc mode
    m_mc = build_mc_sparse(dataset)
    m_vb = build_vb(dataset)
    m_vb[:] = vb_param

    m_mc.kern[:] = m_vb.kern[:]

    m_mc.Z[:] = m_vb.Z[:]
    m_mc.Z.fix()
    L = GPy.util.choleskies.flat_to_triang(m_vb.q_u_chol)[0, :, :]
    u_sample = np.dot(L, np.random.randn(m_vb.num_inducing))
    u_sample += m_vb.q_u_mean.flatten()
    L = GPy.util.linalg.jitchol(m_mc.kern.K(m_mc.Z))
    v_sample, _ = GPy.util.linalg.dtrtrs(L, u_sample)
    m_mc.V[:] = v_sample.reshape(-1, 1)
    return m_mc[:]


@TG
def run_hmc(dataset, x0, N, num_inducing):
    m = build_mc_sparse(dataset)
    m[:] = x0

    def f(x):
        return (-a for a in m._objective_grads(x))

    _, xopt, m_opt = AHMC(f, m.optimizer_array, 20, 40)

    samples, rate = HMC(f, N,
                        Lmax=xopt[1], epsilon=np.exp(xopt[0]),
                        x0=m.optimizer_array)
    return samples


@TG
def mc_predict(dataset, param, samples):
    m = build_mc_sparse(dataset)
    m[:] = param
    ps = []
    Xtest = dataset['Xtest']
    for s in samples:
        m.optimizer_array = s
        ps.append(m.likelihood.predictive_mean(*m.predict_raw(Xtest)))
    return np.mean(ps, 0).reshape(-1, 1)


@TG
def vb_predict(dataset, params):
    m = build_vb(dataset)
    m[:] = params
    return m.predict(dataset['Xtest'])[0]


@TG
def log_prob(p, dataset):
    return np.where(dataset['Ytest'].flatten() == 1, np.log(p.flatten()), np.log(1 - p.flatten())).mean()


class Experiment:
    def __init__(self, seed, Ntrain,  num_inducing, num_samples, fixZ):
        self.seed, self.Ntrain,  self.num_inducing, self.num_samples, self.fixZ \
            = seed, Ntrain, num_inducing, num_samples, fixZ
        np.random.seed(seed)

        self.data = load_data(seed, Ntrain, 'image', num_inducing)

        self.param_vb = optimize_vb(self.data, fixZ)
        self.param_mc = init_mc_from_vb(self.data, self.param_vb)

        self.samples = run_hmc(self.data, self.param_mc,
                               num_samples, num_inducing)

        self.pred_mc = mc_predict(self.data, self.param_mc, self.samples[300::3])
        self.pred_vb = vb_predict(self.data, self.param_vb)

        self.log_pred_mc = log_prob(self.pred_mc, self.data)
        self.log_pred_vb = log_prob(self.pred_vb, self.data)

        # construct models for inspection
        # try:
        #     self.m_vb = build_vb(self.data)
        #     self.m_mc = build_mc_sparse(self.data)
        #     import jug
        #     self.m_vb[:] = jug.value(self.param_vb)
        #     self.m_mc[:] = jug.value(self.param_mc)
        # except:
        #     pass


num_samples = 2000
Ms = [5, 10, 20, 50, 100]
seeds = range(10)
experiments = [
    Experiment(seed, Ntrain=1000, num_inducing=M,
               num_samples=num_samples, fixZ=fixZ)
    for seed in seeds for M in Ms for fixZ in (True, False)]

# plotting
# import pandas as pd
# df = pd.DataFrame()
# for e in experiments:
    # df= df.append(jug.value(e.__dict__))
# df.boxplot(by=['fixZ', 'num_inducing'], column='log_pred_mc')
