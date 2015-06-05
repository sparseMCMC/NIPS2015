import numpy as np
from scipy.linalg import blas
import GPy

class GPMCMC(GPy.core.Model):
    def __init__(self, X, Y, kernel, likelihood, mean_function=None, num_latent_functions = None, name='gpmcmc'):
        GPy.core.Model.__init__(self, name)
        self.X, self.Y, self.kern, self.likelihood = GPy.core.ObsAr(X), GPy.core.ObsAr(Y), kernel, likelihood

        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)

        self.num_data, self.input_dim = X.shape
        num_dataY, self.output_dim = Y.shape
        if num_latent_functions is not None:
            self.output_dim = num_latent_functions
        assert num_dataY == self.num_data

        #add a parameter to represent the whitened variables
        self.V = GPy.core.Param('V', np.random.randn(self.num_data, self.output_dim))
        self.link_parameter(self.V)

    def log_likelihood(self):
        return self._loglik

    def parameters_changed(self):
        K = self.kern.K(self.X)
        self.L = L = GPy.util.linalg.jitchol(K)
        F = self.F = blas.dtrmm(1.0,L, self.V, lower=1, trans_a=0)

        #compute the log likelihood
        self._loglik = self.likelihood.logpdf(F, self.Y).sum()
        dL_dF = self.likelihood.dlogpdf_df(F, self.Y)
        self.likelihood.gradient = self.likelihood.dlogpdf_dtheta(F, self.Y).sum(1).sum(1)

        #here's the prior for V
        self._loglik += -0.5*self.num_data*self.output_dim*np.log(2*np.pi) - 0.5*np.sum(np.square(self.V))

        #set all gradients to zero, then only compute necessary gradients.
        self.gradient = 0.

        #compute dL_dV
        if not self.V.is_fixed:
            self.V.gradient = -self.V + blas.dtrmm(1.0, L, dL_dF, trans_a=1, lower=1)

        #compute kernel gradients
        if not self.kern.is_fixed:
            dL_dL = np.dot(dL_dF, self.V.T) # where the first L is the likelihood, the second L is the triangular matrix
            #nasty reverse Cholesky to get dL_dK
            dL_dK = GPy.util.choleskies.backprop_gradient(dL_dL, L)
            self.kern.update_gradients_full(dL_dK, self.X)

    def predict_raw(self, Xnew, full_cov=False):
        #make a prediction based on the current state of the model
        Kxn = self.kern.K(self.X, Xnew)
        tmp, _ = GPy.util.linalg.dtrtrs(self.L, Kxn)
        mu = np.dot(tmp.T, self.V)
        if full_cov:
            v = self.kern.K(Xnew) - tmp.T.dot(tmp)
        else:
            v = self.kern.Kdiag(Xnew) - np.sum(np.square(tmp),0)
            v = v.reshape(-1,1)
        return mu, v



class SGPMCMC(GPy.core.Model):
    def __init__(self, X, Y,Z, kernel, likelihood, mean_function=None, num_latent_functions=None, name='sgpmcmc'):
        GPy.core.Model.__init__(self, name)
        self.X, self.Y, self.kern, self.likelihood = GPy.core.ObsAr(X), GPy.core.ObsAr(Y), kernel, likelihood
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)

        self.num_data, self.input_dim = X.shape
        num_dataY, self.output_dim = Y.shape
        assert num_dataY == self.num_data

        #add Z as a parameter
        self.Z = GPy.core.Param('Z',Z)
        self.num_inducing, Zinput_dim = self.Z.shape
        assert Zinput_dim == self.input_dim
        self.link_parameter(self.Z)

        if num_latent_functions is None:
            num_latent_functions = Y.shape[1]
        self.num_latent_functions = num_latent_functions

        #add a parameter to represent the whitened variables
        self.V = GPy.core.Param('V', np.random.randn(self.num_inducing, num_latent_functions))
        self.link_parameter(self.V)


    def log_likelihood(self):
        return self._loglik

    def parameters_changed(self):
        Kuu = self.kern.K(self.Z)
        #Kuui, Lu, Lui, logdet_u= GPy.util.linalg.pdinv(Kuu)
        self.Lu = Lu = GPy.util.linalg.jitchol(Kuu)
        logdet_u = 2.*np.sum(np.log(np.diag(Lu)))
        #not actually needed: U = np.dot(Lu, self.V)
        #needed for prediction:
        self.Liv = Liv = GPy.util.linalg.dtrtrs(Lu, self.V, lower=1, trans=1)[0]

        Kfu = self.kern.K(self.X, self.Z)
        #A = np.dot(Lui, Kfu.T).T
        A = GPy.util.linalg.dtrtrs(Lu, Kfu.T, lower=1, trans=0)[0].T
        f_mu = np.dot(A, self.V) # the mean of p(f|u)
        f_var = self.kern.Kdiag(self.X)[:,None] - np.sum(np.square(A),1)[:,None]
        f_var = np.repeat(f_var, self.num_latent_functions, axis=1) # repeat for all indeoendent outputs.

        loglik, dL_dmu, dL_dfvar, dL_dtheta = self.likelihood.variational_expectations(self.Y, f_mu, f_var)

        #compute the log likelihood
        self._loglik = loglik.sum()
        if self.likelihood.size and not self.likelihood.is_fixed: self.likelihood.gradient = dL_dtheta.sum(1).sum(1)

        #here's the prior for V
        self._loglik += -0.5*self.num_inducing*self.num_latent_functions*np.log(2*np.pi) - 0.5*np.sum(np.square(self.V))

        #compute dL_dV
        self.V.gradient = -self.V + np.dot(A.T, dL_dmu)

        #see if we need derivatives via kernel matrices, only compute if needed
        self.kern.gradient = 0.
        self.Z.gradient = 0.
        if not (self.kern.is_fixed and self.Z.is_fixed):
            Adv = A.T*dL_dfvar.sum(1) # A^T * diag(dL_dv)
            Admu = np.dot(A.T, dL_dmu)
            tmp = GPy.util.linalg.dtrtrs(Lu, Adv, lower=1, trans=1)[0]
            #tmp = np.dot(Lui.T, Adv)
            dL_dKuf = np.dot(Liv, dL_dmu.T) - 2.*tmp
            dL_dL = -np.dot( Admu, Liv.T) + 2.*np.dot(A.T, tmp.T)
            dL_dK = GPy.util.choleskies.backprop_gradient(dL_dL.T, Lu)

            if not self.kern.is_fixed:
                self.kern.update_gradients_full(dL_dK, self.Z)
                g = self.kern.gradient[:].copy()
                self.kern.update_gradients_full(dL_dKuf.T, self.X, self.Z)
                g += self.kern.gradient[:].copy()
                self.kern.update_gradients_diag(dL_dfvar.sum(1), self.X)
                self.kern.gradient[:] += g

            if not self.Z.is_fixed:
                self.Z.gradient = self.kern.gradients_X(dL_dK, self.Z)
                self.Z.gradient += self.kern.gradients_X(dL_dKuf, self.Z, self.X)

    def predict_raw(self, Xnew, full_cov=False):
        #make a prediction based on the current state of the model
        Kxu = self.kern.K(Xnew, self.Z)
        mu = np.dot(Kxu, self.Liv)
        tmp = GPy.util.linalg.dtrtrs(self.Lu, Kxu.T, lower=1, trans=0)[0]
        if full_cov:
            v = self.kern.K(Xnew) - tmp.T.dot(tmp)
        else:
            v = self.kern.Kdiag(Xnew) - np.sum(np.square(tmp),0)
            v = v.reshape(-1,1)
        return mu, v




def backprop_chol_grad(dL, L):
    """
    Given the derivative of an objective fn with respect to the cholesky L,
    compute the derivate with respect to the original matrix K, defined as

        K = LL^T

    where L was obtained by Cholesky decomposition
    """

    from scipy import weave
    dL_dK = np.tril(dL).copy()
    code = """
    for(int k=N-1;k>-1;k--){
        //printf("%d\\n",k);
        for(int j=k+1;j<N;j++){
            for(int i=j;i<N; i++){
                //printf("->%d%d\\n",i,j);
                dL_dK(i, k) -= dL_dK(i, j) * L(j, k);
                dL_dK(j, k) -= dL_dK(i, j) * L(i, k);
            }
        }
        for( int j=k+1;j<N; j++){
            //printf("-->%d\\n",j);
            dL_dK(j, k) /= L(k, k);
            dL_dK(k, k) -= L(j, k) * dL_dK(j, k);
        }
        dL_dK(k, k) /= (2 * L(k, k));
    }
    """
    N = L.shape[0]
    weave.inline(code, ['dL_dK', 'L', 'N'], type_converters=weave.converters.blitz)
    return dL_dK


if __name__=='__main__':
    X = np.random.randn(1000,1)
    Y = np.where((np.sin(2*X) + np.random.randn(*X.shape)*0.5) >0, 1,0)
    Z = np.random.randn(20,1)

    k = GPy.kern.RBF(1) + GPy.kern.White(1, 0.01)
    lik = GPy.likelihoods.Bernoulli()

    m = SGPMCMC(X, Y,Z, k, lik)

    pr = GPy.priors.Gamma(3, 0.5)
    m.kern.set_prior(pr)


