import numpy as np
import GPy
from mcmcGP import BinnedPoisson, GPMCMC, SGPMCMC, HMC, AHMC
from itertools import product
from IPython import embed
from scipy.cluster.vq import kmeans
import pyhmc

def load_pines():
    X = np.load('pines.np')
    return X
    
def build_vb(initialZ, binMids, binSize, counts, seed):
    np.random.seed(seed)
    lik = BinnedPoisson(binSize)
    kern = getKern(False)
    return GPy.core.SVGP(X=binMids, Y=counts.reshape(-1,1), Z=initialZ, kernel=kern, likelihood=lik)
 
def optimize_vb(m, max_iters=1000):
    m.Z.fix()
    m.kern.fix()
    m.optimize('bfgs', max_iters=max_iters, messages=True)
    m.Z.unfix()
    m.kern.constrain_positive()
    m.optimize('bfgs', max_iters=max_iters, messages=True)
    return m

def get_samples_vb(m, num_samples):
    mu, var = m._raw_predict(m.X)
    samples = np.random.randn(mu.shape[0], num_samples)*np.sqrt(var) + mu
    return samples

def get_samples_mc(m, samples, numsamples):
    ms, vs = [],[]
    for s in samples:
        m.optimizer_array = s
        mui, vi = m.predict_raw(m.X)
        vi = np.clip(vi, 0, np.inf)
        ms.append(mui); vs.append(vi)

    samples = np.hstack([np.random.randn(mu.shape[0], numsamples)*np.sqrt(var) + mu for mu, var in zip(ms, vs)])
    return samples

def get_samples_mc_full(m, samples):
    Fs = []
    for s in samples:
        m.optimizer_array = s
        Fs.append(m.F)
    return np.hstack(Fs)

def getPriors():
    return {'rbf_lengthscale': GPy.priors.Gamma(1.75,1.), 'rbf_variance': GPy.priors.Gamma(1.2, 1) }

def getKern(isBayesian):
    kern = GPy.kern.RBF(2, lengthscale=1.)+GPy.kern.White(2, 1e-2)
    priors = getPriors()
    if isBayesian:
        kern.rbf.lengthscale.set_prior(priors['rbf_lengthscale'])
        kern.rbf.variance.set_prior(priors['rbf_variance'])
        kern.white.variance.fix(1e-6)
    return kern

def build_mc_sparse(initialZ, binMids, binSize, counts, seed):

    kern = getKern(True)
    lik = BinnedPoisson(binSize)

    return SGPMCMC(binMids, Y=counts.reshape(-1,1), Z=initialZ, kernel=kern, likelihood=lik)

def build_mc_exact( binMids, binSize, counts, seed):

    kern = getKern(False)
    lik = BinnedPoisson(binSize)

    return GPMCMC( X = binMids,  Y=counts.reshape(-1,1), kernel = kern, likelihood = lik ) 

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
    m_mcmc.V[:] = v_sample.reshape(-1,1)
    return m_mcmc

def init_exact_mc_model_from_vb( m_mcmc_exact, m_vb ):
    #This should speed things up a bit.
    m_mcmc_exact.kern[:] = m_vb.kern[:]
    function_sample = get_samples_vb( m_vb, 1).flatten()
    L =  GPy.util.linalg.jitchol(m_mcmc_exact.kern.K(m_mcmc_exact.X))
    v_sample , _ =  GPy.util.linalg.dtrtrs(L, function_sample)
    m_mcmc_exact.V[:] = v_sample.reshape(-1,1)
    return m_mcmc_exact
    
def convertData(X, binsPerDimension):
    Y = np.histogramdd( X, bins = (binsPerDimension,binsPerDimension), range =  ( (0, 1.) , (0., 1.) ) )
    return Y[0].reshape( Y[0].shape[0] * Y[0].shape[1] ) 

def getInitialInducingGrid( nInducingPoints ):
    assert( np.sqrt( nInducingPoints ) == np.floor( np.sqrt( nInducingPoints ) ) ) # check nInducingPoints is a square number.
    sqrtNInducingPoints  = int( np.floor( np.sqrt( nInducingPoints ) ) )
    return getGrid( sqrtNInducingPoints )[1]

def getGrid( nPointsPerDim ):
    linearValues = np.linspace( 0., 1., nPointsPerDim+1 ) 
    binEdges = np.array( [ np.array( elem ) for elem in product(linearValues,linearValues) ] ) 
    offsetValues = linearValues[:-1] + 0.5*np.diff( linearValues )[0]
    binMids =  np.array( [ np.array( elem ) for elem in product(offsetValues,offsetValues) ] ) 
    return binEdges*10., binMids*10. 

def run_hmc(m, N, epsilon, Lmax):
    def f(x):
        return (-a for a in m._objective_grads(x))
    samples, rate = HMC(f, N, Lmax=Lmax, epsilon=epsilon, x0=m.optimizer_array, verbose=True)
    return samples

def priorSample():
    binsPerDimension = 32
    num_samples = 5
    bin_edges, bin_mids = getGrid( binsPerDimension )
    
    np.random.seed(1)
    #There is almost certainly a better way to do this but tempus fugit.
    priors = getPriors()
    kern = getKern(True)
    binArea = np.square( (bin_edges[0,1] - bin_edges[1,1] ) )

    from matplotlib import pyplot as plt
    
    for sampleIndex in range(num_samples):
        print "\n sample index ", sampleIndex, "\n"
        kern.rbf.lengthscale = priors['rbf_lengthscale'].rvs(1)
        kern.rbf.variance = priors['rbf_variance'].rvs(1)
        kern.bias.variance = priors['bias_variance'].rvs(1)
        L = GPy.util.linalg.jitchol(kern.K(bin_mids))
        functionSample = np.dot(L, np.random.randn( bin_mids.shape[0] ) )
        intensities = np.exp( functionSample )
        countSample = np.random.poisson( intensities * binArea )
        print "Total counts ", np.sum( countSample )
        squareIntensities = intensities.reshape( (binsPerDimension, binsPerDimension ))
        squareCounts = countSample.reshape( (binsPerDimension, binsPerDimension ))
        plt.figure()
        plt.imshow( squareCounts, interpolation='nearest')
        plt.title( "Prior sample "+ str(sampleIndex) )
        plt.colorbar()
        plt.figure()
        plt.imshow( squareIntensities, interpolation='nearest')
        plt.colorbar()
        plt.title( "Prior sample "+ str(sampleIndex) )


class Experiment:
    def __init__(self, seed, binsPerDimension , num_inducing, num_samples, vb_iterations, isExact=False):
        self.seed, self.binsPerDimension, self.num_inducing, self.num_samples, self.isExact = seed, binsPerDimension, num_inducing, num_samples, isExact
        np.random.seed(seed)
        
        X = load_pines()
        
        #will need to change bins to be two dimensional.

        self.Y = convertData(X, binsPerDimension)
        binEdges, bin_mids = getGrid( binsPerDimension )
        
        initialZ = getInitialInducingGrid( num_inducing )
        
        #setup and optimize VB model.
        binArea = np.square( (binEdges[0,1] - binEdges[1,1] ) )
        
        if not(isExact):
            self.m_vb = build_vb(initialZ, bin_mids, binArea , self.Y, seed)        
            self.m_vb = optimize_vb(self.m_vb,vb_iterations)
            self.fsamples_vb = get_samples_vb(self.m_vb, 1000)
            self.m_mc = build_mc_sparse(initialZ, bin_mids, binArea, self.Y, seed)
            self.m_mc = init_mc_model_from_vb(self.m_mc, self.m_vb)
            self.samples = run_hmc(self.m_mc, num_samples, 0.125, Lmax = 20)
            self.fsamples_mc = get_samples_mc(self.m_mc, self.samples[50::2], 10)
        else:
            priors = getPriors()
            self.m_mc  = build_mc_exact( bin_mids, binArea, self.Y, seed )
            self.m_mc.kern.rbf.lengthscale.fix(1.)
            self.m_mc.kern.rbf.variance.fix(1.)
            self.m_mc.kern.white.variance.fix(1e-3)
            self.m_mc.optimize('bfgs',messages=True,max_iters = 10000)
            self.m_mc.kern.rbf.lengthscale.constrain_positive()
            self.m_mc.kern.rbf.variance.constrain_positive()
            self.m_mc.kern.white.variance.constrain_positive()
            self.m_mc.kern.rbf.lengthscale.set_prior(priors['rbf_lengthscale'])
            self.m_mc.kern.rbf.variance.set_prior(priors['rbf_variance'])
            self.m_mc.kern.white.variance.fix(1e-3)
            self.samples = run_hmc(self.m_mc, num_samples, epsilon=0.1 , Lmax = 20)
            
#priorSample()    
if __name__ == "__main__": 
    num_samples = 2000
    num_vb_iterations = [10000]

    Ms = [225]
    grids = [64]
    #grids = [32]
    seeds = [0]
    isExact=False
    experiments = [Experiment(seed, binsPerDimension=ng, num_inducing=M, num_samples=num_samples, vb_iterations=vb_iterations, isExact=isExact) for seed in seeds for ng in grids for M in Ms for vb_iterations in num_vb_iterations]

    #from matplotlib import pyplot as plt
    for e in experiments:
        #plt.figure()
        intensities = np.exp(e.fsamples_mc)*e.m_mc.likelihood.binsize
        std = np.std(intensities, axis=1)
        intensities = np.mean(intensities,axis=1)
        squareIntensities = intensities.reshape( (e.binsPerDimension , e.binsPerDimension ))
        #plt.imshow( np.flipud(squareIntensities.T ), interpolation='nearest')
        #plt.colorbar()
        #plt.title( 'Mean posterior intensity')
        np.savetxt( 'intensity_grid%i_M%i_numsamp%i_exact%i.csv'%(e.binsPerDimension, e.num_inducing, e.num_samples, e.isExact),intensities, delimiter=',')
        np.savetxt( 'intensity_std%i_M%i_numsamp%i_exact%i.csv'%(e.binsPerDimension, e.num_inducing, e.num_samples, e.isExact),std, delimiter=',')

        #plt.figure()
        #intensities = np.mean(np.exp(e.fsamples_vb)*e.m_mc.likelihood.binsize,axis=1)
        #squareIntensities = intensities.reshape( (e.binsPerDimension , e.binsPerDimension ))
        #plt.imshow( np.flipud(squareIntensities.T ), interpolation='nearest')
        #plt.colorbar()
        #plt.title( 'Mean posterior intensityi vb')
        #np.savetxt( 'intensity_vb_grid%i_M%i_numsamp%i.csv'%(e.binsPerDimension, e.num_inducing, e.num_samples),intensities, delimiter=',')
