import numpy as np
import mcmcGP
import GPy
from multiclassLikelihood_GPy import Multiclass
from ahmc import HMC, AHMC
import sys
from time import time

np.random.seed(1)
ndata = None # none for all data
vb_filename = 'mnist_467M_scg.pickle'
ahmc_num_steps = 50
ahmc_s_per_step = 20
hmc_num_samples = 300
thin = 2

data = GPy.load('mnist_pickle')
X_train, Y_train, X_test, Y_test = data['Xtrain'], data['Ytrain'], data['Xtest'], data['Ytest']
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#scale data
X_train = X_train/255.0
X_train = X_train*2. - 1.
X_test = X_test/255.0
X_test = X_test*2. - 1.

#randomize order
i = np.random.permutation(X_train.shape[0])
i = i[:ndata]
X_train, Y_train = X_train[i,:], Y_train[i,:]

#load vb model, create mcmc model, init mc v=from vb
m_vb = GPy.load(vb_filename)
k = GPy.kern.RBF(X_train.shape[1], ARD=True) + GPy.kern.White(1, 1e-3)
lik = Multiclass()
m = mcmcGP.SGPMCMC(X=X_train, Y=Y_train, kernel=k.copy(), likelihood=lik.copy(), num_latent_functions=10, Z=m_vb.Z*1)
m.update_model(False)
m.kern[:] = m_vb.kern[:]*1
m.Z.fix()
m.kern.fix()
L = GPy.util.choleskies.flat_to_triang(m_vb.q_u_chol)
U = np.vstack([np.dot(L[i,:,:], np.random.randn(m.V.shape[0])) for i in range(10)]).T
U = U + m_vb.q_u_mean
K = m.kern.K(m.Z)
L = GPy.util.linalg.jitchol(K)
m.V[:] = GPy.util.linalg.dtrtrs(L, U)[0]
m.likelihood.fix(1e-3)
m.update_model(True)


#set a prior
pr = GPy.priors.Gamma(1.,1.)
m.kern.rbf.lengthscale.set_prior(pr)
pr = GPy.priors.Gamma(4.,1.3)
m.kern.rbf.variance.set_prior(pr)
m.kern.white.fix(1e-6)

#samples mcmc
def f(x):
    return (-a for a in m._objective_grads(x))
#_, xopt, m_opt = AHMC(f, m.optimizer_array, ahmc_num_steps, ahmc_s_per_step, verbose=1)
xopt = [np.log(0.01), 30]
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

#from matplotlib import pyplot as plt
#plt.figure()
#plt.plot(p)
#plt.plot(p_vb, '.')
#plt.twinx();plt.plot(Y_test, 'ks')
#



