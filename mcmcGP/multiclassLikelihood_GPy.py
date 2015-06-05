from fastMultiClassLikelihood import likelihoodQuadrature, predictiveQuadrature
import GPy
import numpy as np

class Multiclass(GPy.likelihoods.Likelihood):
    def __init__(self, gp_link=None):
        if gp_link is not None:
            raise ValueError, "this likelihood assumes a robust-max inverse-link"

        super(Multiclass, self).__init__(GPy.likelihoods.link_functions.Identity(), 'Bernoulli')
        self.delta = GPy.core.Param('delta', 0.01)
        self.link_parameter(self.delta)

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        if gh_points is None:
            gh_x, gh_w = self._gh_points()
        else:
            gh_x, gh_w = gh_points
        ret = likelihoodQuadrature(m,v, np.array(Y.flatten(), dtype=np.int), self.delta*1., gh_w, gh_x )
        ret = list(ret)
        ret[-1] = np.array([ret[-1]]).reshape(1,1,1)
        return ret

    def update_gradients(self, g):
        self.delta.gradient = g

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        assert full_cov is False, "cannot make correlated predictions"
        return predictiveQuadrature(mu, var, self.delta*1, *self._gh_points()[::-1] ), None



if __name__=='__main__':

    l = Multiclass()
    l.fix(1e-3)
    X = np.random.randn(1000,1)
    k = GPy.kern.RBF(1) + GPy.kern.White(1, variance=1e-2)
    K = k.K(X)
    F = np.random.multivariate_normal(np.zeros(X.shape[0]), K, 3).T
    Y = np.argmax(F,1).reshape(-1,1)

    m = GPy.core.SVGP(X=X, Y=Y, Z = np.random.randn(10,1), kernel=k, likelihood=l, num_latent_functions=3)
    m.optimize('bfgs', max_iters=50)

    xx =np.linspace(-3,3, 100).reshape(-1,1)
    mu, v = m._raw_predict(xx)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(xx, mu)
    plt.twinx()
    plt.plot(X, F, '.')



