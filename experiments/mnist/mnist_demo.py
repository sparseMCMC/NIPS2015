import numpy as np
import mcmcGP
import GPy
from multiclassLikelihood_GPy import Multiclass
from ahmc import HMC, AHMC
import climin
import sys
from time import time

np.random.seed(0)
reduced_dim = 45
ndata = None # none for all data
num_inducing = 500
ahmc_num_steps = 50
ahmc_s_per_step = 20
hmc_num_samples = 300
vb_frozen_iters = 8000
vb_max_iters = 200000
vb_batchsize = 1000
step_rates = 1e-1, 5e-2
thin = 2

data = GPy.load('mnist_pickle')
X_train, Y_train, X_test, Y_test = data['Xtrain'], data['Ytrain'], data['Xtest'], data['Ytest']
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#normalize dta
#Xmean, Xstd = X_train.mean(0), X_train.std(0)
#Xstd = np.where(Xstd==0, 1,Xstd)
#X_train = (X_train-Xmean)/Xstd
#X_test = (X_test-Xmean)/Xstd

#scale data
X_train = X_train/255.0
X_train = X_train*2. - 1.
X_test = X_test/255.0
X_test = X_test*2. - 1.

#pca data
#X_train, W = GPy.util.linalg.pca(X_train, reduced_dim)
#X_test = np.linalg.lstsq(W.T, X_test.T)[0].T

#randomize order
i = np.random.permutation(X_train.shape[0])
i = i[:ndata]
X_train, Y_train = X_train[i,:], Y_train[i,:]

#obtain k-mean centers for Z init
from scipy.cluster.vq import kmeans
Z, _ = kmeans(X_train[::20,:], num_inducing)
stop


#build vb model
k = GPy.kern.RBF(X_train.shape[1], ARD=True) + GPy.kern.White(1, 1e-3)
lik = Multiclass()
m_vb = GPy.core.SVGP(X=X_train, Y=Y_train, kernel=k.copy(), likelihood=lik.copy(), num_latent_functions=10, Z=Z, batchsize=vb_batchsize)
m = mcmcGP.SGPMCMC(X=X_train, Y=Y_train, kernel=k.copy(), likelihood=lik.copy(), num_latent_functions=10, Z=Z)
m_vb.likelihood.delta.fix(1e-3)
m.likelihood.delta.fix(1e-3)

#set a prior
#pr = GPy.priors.Gamma(3,3)
#m_vb.kern.set_prior(pr)
pr = GPy.priors.Gamma(3,3)
m.kern.set_prior(pr)

#optimize vb
m_vb.Z.fix()
#m_vb.optimize('bfgs', messages=1, max_iters=vb_frozen_iters)
#m_vb.Z.unfix()
#m_vb.optimize('bfgs', messages=1, max_iters=vb_max_iters)
trace_times = []; trace_errs = []; trace_lls = []
class cb():
    def __init__(self, stop_pc=1., holdout_inverval=200, max_iters=np.inf):
        self.stop_pc = stop_pc
        self.holdout_inverval = holdout_inverval
        self.max_iters = max_iters

    def __call__(self, info):
        if (info['n_iter']%self.holdout_inverval)==0 and info['n_iter']>0:
            mu, _ = m_vb.predict(X_test)
            percent = np.mean(np.argmax(mu,1)==Y_test.flatten())
            trace_times.append(time()); trace_lls.append(m_vb.log_likelihood()); trace_errs.append(percent)

            print '\n', m_vb.log_likelihood(), percent
            if percent >= self.stop_pc:
                return True
        else:
            print '\r',m_vb.log_likelihood(),
            sys.stdout.flush()

        if info['n_iter'] >= self.max_iters:
            return True

        return False

stop

m_vb.kern.fix()
m_vb.Z.fix()
opt = climin.Adadelta(m_vb.optimizer_array, m_vb.stochastic_grad, step_rate=step_rates[0])
opt.minimize_until(cb(stop_pc=0.9, max_iters=600))
m_vb.kern.constrain_positive()
m_vb.Z.unfix()
opt = climin.Adadelta(m_vb.optimizer_array, m_vb.stochastic_grad, step_rate=step_rates[1])
opt.minimize_until(cb(max_iters=vb_max_iters))



#set mcmc from vb solution
m.kern[:] = m_vb.kern[:]*1
m.Z[:] = m_vb.Z[:]*1
m.Z.fix()
L = GPy.util.choleskies.flat_to_triang(m_vb.q_u_chol)
U = np.vstack([np.dot(L[:,:,i], np.random.randn(m.V.shape[0])) for i in range(10)]).T
U = U + m_vb.q_u_mean
K = m.kern.K(m.Z)
L = GPy.util.linalg.jitchol(K)
m.V[:] = GPy.util.linalg.dtrtrs(L, U)[0]

#samples mcmc
def f(x):
    return (-a for a in m._objective_grads(x))
_, xopt, m_opt = AHMC(f, m.optimizer_array, ahmc_num_steps, ahmc_s_per_step, verbose=1)
samples, rate = HMC(f, hmc_num_samples, epsilon=np.exp(xopt[0]), Lmax=xopt[1], x0=m.optimizer_array, verbose=1)

#order test points for pretty plotting
i = np.argsort(Y_test.flatten())
X_test = X_test[i]
Y_test = Y_test[i]

ms, vs, ps = [], [], []
for s in samples[::thin]:
    m.optimizer_array = s
    mu, v = m.predict_raw(X_test)
    ms.append(mu);vs.append(v)
    ps.append(m.likelihood.predictive_values(mu, np.tile(v,(1,10)))[0])

p = np.mean(ps,0)
mu_vb, v_vb = m_vb._raw_predict(X_test)
p_vb = m.likelihood.predictive_values(mu_vb, v_vb)[0]

from matplotlib import pyplot as plt
plt.figure()
plt.plot(p)
plt.plot(p_vb, '.')
plt.twinx();plt.plot(Y_test, 'ks')




