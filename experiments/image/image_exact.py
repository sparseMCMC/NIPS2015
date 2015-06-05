import numpy as np
import GPy
from mcmcGP import GPMCMC, HMC, AHMC
from jug import TaskGenerator as TG
from scipy import io

def load_data(seed, ntrain, datasetName):
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
    return dict(Xtrain=xtrain, Ytrain=ytrain, Xtest=xtest, Ytest=ytest)


@TG
def log_prob(p, dataset):
    return np.where(dataset['Ytest'].flatten() == 1, np.log(p.flatten()), np.log(1 - p.flatten())).mean()


def build_mc(dataset):
    X = dataset['Xtrain']

    lik = GPy.likelihoods.Bernoulli()
    kern = GPy.kern.RBF(X.shape[1], lengthscale=1., ARD=True) + GPy.kern.White(1)

    kern.rbf.lengthscale.set_prior(GPy.priors.Gamma(1.1, 1))
    kern.rbf.variance.set_prior(GPy.priors.Gamma(1.1, 1))
    kern.white.fix(1e-6)

    m = GPMCMC(X=X, Y=dataset['Ytrain'], kernel=kern, likelihood=lik)
    return m


@TG
def run_hmc(dataset, N):
    m = build_mc(dataset)

    #with the kernel fixed, optimize the latent function to MAP
    m.kern.fix()
    m.optimize('bfgs',messages=True,max_iters = 1000)
    m.kern.rbf.constrain_positive()

    def f(x):
        return (-a for a in m._objective_grads(x))

    _, xopt, m_opt = AHMC(f, m.optimizer_array, 20, 20, L_bounds=[20,30], verbose=True)

    samples, rate = HMC(f, N,
                        Lmax=xopt[1], epsilon=np.exp(xopt[0]),
                        x0=m.optimizer_array)
    return samples


@TG
def predict(dataset, samples):
    m = build_mc(dataset)
    ps = []
    Xtest = dataset['Xtest']
    for s in samples:
        m.optimizer_array = s
        ps.append(m.likelihood.predictive_mean(*m.predict_raw(Xtest)))
    return np.mean(ps, 0).reshape(-1, 1)



class Experiment:
    def __init__(self, seed, Ntrain,  num_samples):
        self.seed, self.Ntrain,  self.num_samples = seed, Ntrain, num_samples
        np.random.seed(seed)

        self.data = load_data(seed, Ntrain, 'image')

        self.samples = run_hmc(self.data, num_samples)

        self.pred_mc = predict(self.data, self.samples[300::3])
        self.log_pred_mc = log_prob(self.pred_mc, self.data)


num_samples = 5000
seeds = range(10)
experiments = [Experiment(seed, Ntrain=1000, num_samples=num_samples)
               for seed in seeds]

