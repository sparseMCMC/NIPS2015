#Timing calculations for pines experiment.

#Be careful before running this to "export OMP_NUM_THREADS=1" as otherwise the comparison may not be fair.

import pyhmc
import numpy as np
from spatial_demo import *
import timeit
from IPython import embed

def build_reference_experiment_model( isVB, binsPerDimension ):
    num_inducing = 225
    seed = 1
    X = load_pines()
    Y = convertData(X, binsPerDimension)
    binEdges, bin_mids = getGrid( binsPerDimension )
    initialZ = getInitialInducingGrid( num_inducing )
    binArea = np.square( (binEdges[0,1] - binEdges[1,1] ) )
    
    if isVB:
        m_vb = build_vb(initialZ, bin_mids, binArea , Y, seed)        
        m_vb = optimize_vb(m_vb,10)
        model = build_mc_sparse(initialZ, bin_mids, binArea, Y, seed)
        model = init_mc_model_from_vb(model, m_vb)
    else:
        priors = getPriors()
        model = build_mc_exact( bin_mids, binArea, Y, seed )
        model.kern.rbf.lengthscale.fix(1.)
        model.kern.rbf.variance.fix(1.)
        model.kern.white.variance.fix(1e-3)
        model.optimize('bfgs',messages=True,max_iters = 100)
        model.kern.rbf.lengthscale.constrain_positive()
        model.kern.rbf.variance.constrain_positive()
        model.kern.white.variance.constrain_positive()
        model.kern.rbf.lengthscale.set_prior(priors['rbf_lengthscale'])
        model.kern.rbf.variance.set_prior(priors['rbf_variance'])
        model.kern.white.variance.fix(1e-3)
    return model

#32 x 32 calculations.
averageLeapFrogSteps = 10
acf_cutoff = 0.05 #Following estimation method of Hoffman and Gelman 2011 "The no U-turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo"
burn_in = 100

full_mcmc_samples_32 = np.loadtxt('long_run_exact_samples.np')
auto_correlation_time_mcmc_full_32 = pyhmc.integrated_autocorr1( full_mcmc_samples_32[burn_in:,:], acf_cutoff ).max()

variational_samples_32 = np.loadtxt('225_inducing_point_samples_32_grid_comma.np',delimiter=',')
auto_correlation_time_variational_32 = pyhmc.integrated_autocorr1( variational_samples_32[burn_in:,:], acf_cutoff ).max()

variational_samples_64 = np.loadtxt('225_inducing_point_samples_64_grid_comma.np',delimiter=',')
auto_correlation_time_variational_64 = pyhmc.integrated_autocorr1( variational_samples_64[burn_in:,:], acf_cutoff ).max()

#Now do the timing bit.

vb_32_model = build_reference_experiment_model( True, 32 )
vb_64_model = build_reference_experiment_model( True, 64 )
mc_32_model = build_reference_experiment_model( False, 32 )

number_of_trials = 10

timer_vb_32 = timeit.Timer(lambda: vb_32_model.parameters_changed() )
timer_vb_64 = timeit.Timer(lambda: vb_64_model.parameters_changed() )
timer_mc_32 = timeit.Timer(lambda: mc_32_model.parameters_changed() )

vb_32_time = timer_vb_32.timeit(number_of_trials)/float(number_of_trials)
vb_64_time = timer_vb_64.timeit(number_of_trials)/float(number_of_trials)
mc_32_time = timer_mc_32.timeit(number_of_trials)/float(number_of_trials)

print "full_mcmc_samples_32.shape ",full_mcmc_samples_32[burn_in:,:].shape
print "variational_samples_32.shape ", variational_samples_32[burn_in:,:].shape
print "variational_samples_64.shape ", variational_samples_64[burn_in:,:].shape

print "auto_correlation_time_variational_32 ",auto_correlation_time_variational_32
print "auto_correlation_time_variational_64 ",auto_correlation_time_variational_64
print "auto_correlation_time_mcmc_full_32 ",auto_correlation_time_mcmc_full_32

print "Time for vb 32 effective sample ", vb_32_time * auto_correlation_time_variational_32 * averageLeapFrogSteps
print "Time for vb 64 effective sample ", vb_64_time * auto_correlation_time_variational_64 * averageLeapFrogSteps
print "Time for mc_32_time ", mc_32_time * auto_correlation_time_mcmc_full_32 * averageLeapFrogSteps


