import GPy
import mcmcGP
from mcmcGP.multiclassLikelihood_GPy import Multiclass
import cPickle
from IPython import embed
from scipy.cluster.vq import kmeans
from itertools import product
from mcmcGP.ahmc import HMC, AHMC
import numpy as np
from matplotlib import pylab as plt

def trueDensity(X):
    #coordinates are (x,y)
    meanR = np.array( [0. , 0.5] )
    meanG = np.array( [-1.5, 0.] )
    meanB = np.array( [1.5, 0.] )

    covR = np.array( [[ 5., 0], [0., 0.1] ] )
    covG = np.array( [[ 0.05, 0], [0, 3. ] ] )
    transformMatrix = np.array( [[ 1., 1.], [-1., 1.] ] ) / np.sqrt(2.)
    diagonalMatrix = np.diag( [3., 0.1 ] )
    covB = np.dot( transformMatrix.T , np.dot( diagonalMatrix  , transformMatrix ) )

    LR = np.linalg.cholesky( covR )
    LG = np.linalg.cholesky( covG )
    LB = np.linalg.cholesky( covB )

    means = [ meanR, meanG, meanB ]
    Ls = [ LR, LG, LB ]

    #evaluate density for each of the colors
    ds = [GPy.util.linalg.dtrtrs(L, (X-m).T)[0] for L, m in zip(Ls, [meanR,meanG,meanB])]
    log_ps = np.vstack([-np.sum(np.log(np.diag(L))) - 0.5*np.sum(np.square(d),0) for L, d in zip(Ls, ds)]).T
    p = np.exp(log_ps)
    return p/p.sum(1)[:,None]


def getGrid( nPointsPerDim, limitsX, limitsY ):
    linearValuesX = np.linspace( limitsX[0], limitsX[1], nPointsPerDim+1 )
    linearValuesY = np.linspace( limitsY[0], limitsY[1], nPointsPerDim+1 )
    binEdges = np.array( [ np.array( elem ) for elem in product(linearValuesX,linearValuesY) ] )
    offsetValuesX = linearValuesX[:-1] + 0.5*np.diff( linearValuesX )[0]
    offsetValuesY = linearValuesY[:-1] + 0.5*np.diff( linearValuesY)[0]
    binMids =  np.array( [ np.array( elem ) for elem in product(offsetValuesX,offsetValuesY) ] )
    return binEdges, binMids

def drawPredictionSpace(  model ):
    #plt.figure()
    nGridPoints = 200
    predictionPoints = getGrid( nGridPoints, [-6.,6.],[-6.,6.] )[1]
    predictions = np.array(model.predict( predictionPoints )[0])
    redChannel = predictions[:,0].reshape(nGridPoints,nGridPoints)
    greenChannel = predictions[:,1].reshape(nGridPoints,nGridPoints)
    blueChannel = predictions[:,2].reshape(nGridPoints,nGridPoints)
    #plt.imshow( np.flipud( np.array( [ redChannel, blueChannel, greenChannel ] ).T ) )

    plt.figure()
    levels = [0.35, 0.97, 0.991]
    for level in levels: # loop helps matplotlib2tikz
        plt.contour(predictionPoints[:,0].reshape(nGridPoints, nGridPoints), predictionPoints[:,1].reshape(nGridPoints, nGridPoints), redChannel, levels=[level], colors='r', lw=3)
        plt.contour(predictionPoints[:,0].reshape(nGridPoints, nGridPoints), predictionPoints[:,1].reshape(nGridPoints, nGridPoints), blueChannel, levels=[level], colors='g', lw=3)
        plt.contour(predictionPoints[:,0].reshape(nGridPoints, nGridPoints), predictionPoints[:,1].reshape(nGridPoints, nGridPoints), greenChannel, levels=[level], colors='b', lw=3)

    plt.plot(model.Z[:,0], model.Z[:,1], 'ko', mew=0, ms=5)
    for i,col in enumerate('rbg'):
        ii = model.Y.flat==i
        plt.plot(model.X[ii,0], model.X[ii,1],col+'o', mew=0, ms=6, alpha=0.5)
    plt.xlim(-6,6)
    plt.ylim(-5,5)

    #get true density and contour
    #true_p = trueDensity(predictionPoints)
    #for p,col in zip(true_p.T, 'rbg'):
       #plt.contour(predictionPoints[:,0].reshape(nGridPoints, nGridPoints), predictionPoints[:,1].reshape(nGridPoints, nGridPoints), p.reshape(nGridPoints, nGridPoints), levels=[0.5, 0.95], colors=col, lw=2, ls='--')

def samplingPredict( X_test, m_mc, n_thin, samples ):
    ms, vs, ps = [], [], []
    for s in samples[::n_thin]:
        m_mc.optimizer_array = s
        mu, v = m_mc.predict_raw(X_test)
        ms.append(mu);vs.append(v)
        ps.append(m_mc.likelihood.predictive_values(mu, np.tile(v,(1,10)))[0])
    p = np.mean(ps,0)
    return tuple([p])

data = cPickle.load( open('data','r') )

np.random.seed(1)
num_inducing = 50
#ahmc_num_steps = 50
#ahmc_s_per_step = 20
hmc_num_samples = 10000
vb_frozen_iters = 100
vb_max_iters = 2000
n_thin = 500

num_latent_functions = 3

kernel = GPy.kern.RBF(data['X_train'].shape[1], ARD=False) + GPy.kern.White(1, 1e-3)
lik = Multiclass()

Z, _ = kmeans(data['X_train'], num_inducing)

m_vb = GPy.core.SVGP(X=data['X_train'], Y=data['Y_train'].reshape(-1,1), kernel=kernel.copy(), likelihood=lik.copy(), num_latent_functions=num_latent_functions, Z=Z)
m_mc = mcmcGP.SGPMCMC(X=data['X_train'], Y=data['Y_train'].reshape(-1,1), kernel=kernel.copy(), likelihood=lik.copy(), num_latent_functions=num_latent_functions, Z=Z)

pr = GPy.priors.Gamma(3,3)
m_mc.kern.rbf.variance.set_prior(pr)
#m_mc.kern.white.variance.set_prior(pr)
m_mc.kern.white.variance.fix(1e-3)
pr = GPy.priors.Gamma(3,2)
m_mc.kern.rbf.lengthscale.set_prior(pr)

m_vb.likelihood.delta.fix(1e-3)
m_mc.likelihood.delta.fix(1e-3)

m_vb.Z.fix()
m_vb.kern.fix()

#m_vb.optimize('bfgs', max_iters=vb_frozen_iters, messages=True)
#m_vb.Z.unfix()
#m_vb.kern.constrain_positive()
#m_vb.kern.white.fix(1e-3) # to keep the same as mcmc
#m_vb.optimize('bfgs', max_iters=vb_max_iters, messages=True)
m_vb = GPy.load('m_vb')

drawPredictionSpace( m_vb )
from matplotlib2tikz import save as save_tikz
save_tikz('simple_vb.tikz')

#m_mc.kern.rbf[:] = m_vb.kern.rbf[:]*1
m_mc.Z[:] = m_vb.Z[:]*1
m_mc.Z.fix()
L = GPy.util.choleskies.flat_to_triang(m_vb.q_u_chol)
U = np.vstack([np.dot(L[i,:,:], np.random.randn(m_mc.V.shape[0])) for i in range(3)]).T
U = U + m_vb.q_u_mean
K = m_mc.kern.K(m_mc.Z)
L = GPy.util.linalg.jitchol(K)
m_mc.V[:] = GPy.util.linalg.dtrtrs(L, U)[0]

#samples mcmc
def f(x):
    return (-a for a in m_mc._objective_grads(x))
#m_mc.kern.fix()
#m_mc.optimize('scg', max_iters=90) # get a little close to the mode...
#m_mc.kern.constrain_positive()

#samples, rate = HMC(f, hmc_num_samples, epsilon=0.008, Lmax=30, x0=m_mc.optimizer_array, verbose=1)
samples = np.load('samples')

m_mc.predict = lambda x : samplingPredict( x, m_mc, n_thin, samples )

drawPredictionSpace( m_mc )
save_tikz('simple_mc.tikz')


#embed()
