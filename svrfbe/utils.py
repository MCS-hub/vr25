# math
import numpy as np
from scipy.stats import ortho_group, uniform
from scipy.linalg import sqrtm, cho_factor, cho_solve
from scipy.stats import multivariate_normal
import jax
import ot

import numpy.random as nr
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from scipy.stats import norm

randtoolbox = rpackages.importr("randtoolbox")


# use Cholesky decomposition to obtain the inverse
# gpt's solution
def cho_inv(A, dim):
    factor = cho_factor(A)
    return cho_solve(factor, np.eye(dim))


def gen_matrix_with_eigs(eigs):
    """
    Generates a symmetric matrix with eigenvalues `eigs`.
    """
    dim = len(eigs)
    x = ortho_group.rvs(dim)
    return x.T @ np.diag(eigs) @ x


def gen_matrix_with_eigbounds(low, high, dim: int):
    """
    Generates a symmetric matrix with eigenvalues within [low, high].
    """
    eigs = low + (high - low) * uniform.rvs(size=dim)
    return gen_matrix_with_eigs(eigs)


def clip_matrix(M: np.array, upper):
    """
    Assuming that M is PSD, return the matrix obtained by clipping the singular values at upper.
    """
    eigs = np.linalg.eig(M)
    return eigs[1] @ np.diag(np.minimum(eigs[0], upper)) @ eigs[1].T


def wasserstein_dist(mu_0, mu_1, Sigma_0, Sigma_1):
    """
    Return the squared W2 distance between N(mu_0, Sigma_0), N(mu_1, Sigma_1).
    """
    half_0 = sqrtm(Sigma_0)
    half_1 = sqrtm(Sigma_1)
    return np.sum((mu_0 - mu_1) ** 2) + np.sum((half_0 - half_1) ** 2)


def H(mu, Sigma):
    """
    Return the negentropy of N(mu, Sigma).
    """
    d = len(mu)
    assert d == Sigma.shape[0] and d == Sigma.shape[1]
    # return -(d / 2 * (1 + np.log(2 * np.pi)) + 1 / 2 * np.log(np.linalg.det(Sigma)))
    return -(d / 2 * (1 + np.log(2 * np.pi)) + 1 / 2 * np.linalg.slogdet(Sigma)[1])


def kl_divergence(mu_0, mu_1, Sigma_0, Sigma_1):
    """
    Return the KL divergence
    KL( N(mu_0, Sigma_0) || N(mu_1, Sigma_1) ).
    """
    d = mu_0.shape[0]

    # det_0 = np.linalg.det(Sigma_0)
    # det_1 = np.linalg.det(Sigma_1)
    Sigma_1_inv = cho_inv(Sigma_1, d)
    # div = np.log(det_1) - np.log(det_0)
    div = np.linalg.slogdet(Sigma_1)[1] - np.linalg.slogdet(Sigma_0)[1]
    div -= d
    div += (Sigma_1_inv * Sigma_0).sum()
    div += (mu_1 - mu_0).T @ Sigma_1_inv @ (mu_1 - mu_0)
    div /= 2
    return div


def softmax(x):
    """
    Univariate softmax function.
    """
    return np.exp(x) / (1 + np.exp(x))


def kl_divergence_given_target(mu_G, Sigma_G, target_log_pdf, num_samples=1000):

    samples = np.random.multivariate_normal(mu_G, Sigma_G, num_samples)
    logp_G = multivariate_normal(mean=mu_G, cov=Sigma_G).logpdf(samples)
    log_p_T = target_log_pdf(samples)
    kl_div = np.mean(logp_G - log_p_T)

    return kl_div


def sinkhorn_divergence(
    num_samples=1000,
    reg=1e-1,
    repeat=10,
    samples1=None,
    mu1=None,
    Sigma1=None,
    mu2=None,
    Sigma2=None,
):
    results = []
    for _ in range(repeat):
        if samples1 is None:
            assert mu1 is not None
            assert Sigma1 is not None
            samples1 = np.random.multivariate_normal(mu1, Sigma1, num_samples)

        if mu1 is None or Sigma1 is None:
            assert samples1 is not None
            indexes = np.random.choice(samples1.shape[0], num_samples)
            samples1 = samples1[indexes, :]
        samples2 = np.random.multivariate_normal(mu2, Sigma2, num_samples)
        results.append(
            ot.bregman.empirical_sinkhorn_divergence(samples1, samples2, reg)
        )
    return results


class Model:
    def __init__(self, log_density_class):
        # Initialize with an instance of a log density class (like eight_schools_centered)
        self.log_density_class = log_density_class

        # Define the gradient of the log-density function
        self.grad_log_density = jax.grad(self.log_density_class.log_density)

        # Define the Hessian of the log-density function
        self.hessian_log_density = jax.hessian(self.log_density_class.log_density)
        self.dim = self.log_density_class.D

    def log_density(self, x):
        # Compute the log density using the provided class's method
        return self.log_density_class.log_density(x)

    def log_density_gradient(self, x):
        # Compute the gradient of the log density at x
        return self.grad_log_density(x)

    def log_density_hessian(self, x):
        # Compute the Hessian of the log density at x
        return self.hessian_log_density(x)


def random_sequence_rqmc(dim, i=0, n=1, random_seed=0):
    """
    Generate uniform RQMC random sequence
    Reference: Quasi-Monte Carlo Variational Inference, Buchholz et al., 2018
    """
    u = np.array(
        randtoolbox.sobol(n=n, dim=dim, init=(i == 0), scrambling=1, seed=random_seed)
    ).reshape((n, dim))
    # randtoolbox for sobol sequence
    return u


def transform_uniform_to_normal(u, mu, sigma):
    """
    Generate a multivariate normal based on
    a uniform sequence
    """
    l_cholesky = np.linalg.cholesky(sigma)
    epsilon = norm.ppf(u).transpose()
    res = np.transpose(l_cholesky.dot(epsilon)) + mu
    return res
