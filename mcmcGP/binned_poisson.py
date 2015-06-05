from __future__ import division

import numpy as np
from scipy import stats,special
import scipy as sp
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood

class BinnedPoisson(Likelihood):
    def __init__(self, binsize=1.,  gp_link=None):
        self.binsize=binsize
        self.logbinsize = np.log(self.binsize)
        if gp_link is None:
            gp_link = link_functions.Log()
        super(BinnedPoisson, self).__init__(gp_link, name='Poisson')

    def _conditional_mean(self, f):
        return self.gp_link.transf(gp) * self.binsize

    def logpdf(self, f, y):
        return (f + self.logbinsize)*y - np.exp(f)*self.binsize - special.gammaln(y+1)
    def dlogpdf_df(self, f, y):
        return y - np.exp(f)*self.binsize

    def pdf_link(self, link_f, y, Y_metadata=None):
        return stats.poisson.pmf(y,link_f*self.binsize)

    def logpdf_link(self, link_f, y, Y_metadata=None):
        return -link_f*self.binsize + y*np.log(link_f*self.binsize) - special.gammaln(y+1)

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        return y/link_f/self.binsize - 1

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        return -y/(link_f*self.binsize)**2

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        return 2.*y/(link_f*self.binsize)**3

    def conditional_mean(self,gp):
        return self.gp_link.transf(gp)*self.binsize

    def conditional_variance(self,gp):
        return self.gp_link.transf(gp)*self.binsize

    def samples(self, gp, Y_metadata=None):
        orig_shape = gp.shape
        gp = gp.flatten()
        Ysim = np.random.poisson(self.gp_link.transf(gp)*self.binsize)
        return Ysim.reshape(orig_shape)

    def variational_expectations(self, Y, m, v, Y_metadata=None):
        """
        Use Gauss-Hermite Quadrature to compute

           E_p(f) [ log p(y|f) ]
           d/dm E_p(f) [ log p(y|f) ]
           d/dv E_p(f) [ log p(y|f) ]

        where p(f) is a Gaussian with mean m and variance v. The shapes of Y, m and v should match.
        """

        #only tractable if link fn is a log, else revert to quadrature:
        if not isinstance(self.gp_link, link_functions.Log):
            return Likelihood.variational_expectations(self, Y, m, v)

        #we have a log link: the expactations are tractable!
        exponent = np.exp(m + v/2.)
        F = -exponent*self.binsize + Y*m - special.gammaln(Y+1) + Y*np.log(self.binsize)
        dF_dm = Y - exponent*self.binsize
        dF_dv = -exponent*self.binsize/2.
        return F, dF_dm, dF_dv, None





