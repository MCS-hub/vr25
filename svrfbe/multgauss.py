# math
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.stats import uniform, multivariate_normal
from scipy.linalg import cholesky, cho_factor, cho_solve
import warnings

# python utils
from functools import partial

from .experiment import Experiment
from .utils import (
    gen_matrix_with_eigs,
    cho_inv,
    kl_divergence,
    wasserstein_dist,
    sinkhorn_divergence,
    random_sequence_rqmc,
    transform_uniform_to_normal,
)


class GaussianExperimentEstimator(Experiment):
    """
    Run convergence experiment for different algorithms on randomly generated target.
    """

    def __init__(
        self, dim=5, alpha=0.01, beta=1, seed=0, estimator="G", W2=False, c=0.9
    ):
        """
        Initialize parameters of experiment:
        - dim, alpha
        - true (mu, Sigma) and precomputed Sigma^{-1}
        - nabla V(x)
        - initial Sigma for experiments
        """

        """
        define potential
        V = 1/2 (x - mu) Sigma^-1 (x - mu)
        """

        if seed is not None:
            np.random.seed(seed)

        def log_potential(x, mu, Sigma_inv):
            assert x.shape[0] == mu.shape[0]
            first = x.T - mu
            second = np.dot(Sigma_inv, (x.T - mu).T)
            return 1 / 2 * np.dot(first, second)

        # randomly initialize mu, Sigma
        mu_true = uniform.rvs(size=(dim,))
        Sigma_true = gen_matrix_with_eigs(np.geomspace(1 / beta, 1 / alpha, dim))
        Sigma_true_inv = cho_inv(Sigma_true, dim)

        log_potential_pure = partial(
            log_potential, mu=mu_true, Sigma_inv=Sigma_true_inv
        )

        jax_mu_true = jnp.array(mu_true)
        jax_Sigma_true_inv = jnp.array(Sigma_true_inv)

        def jax_lp(x):
            first = x - jax_mu_true  # shape (N, D)
            temp = -0.5 * jnp.einsum("ij,jk,ik->i", first, jax_Sigma_true_inv, first)
            return temp.squeeze()

        if estimator == "G":

            def gradient_oracle(mu, Sigma):
                nabla_1 = Sigma_true_inv @ (mu - mu_true)
                nabla_2 = Sigma_true_inv
                return nabla_1, nabla_2

        elif estimator == "SG":

            def gradient_oracle(mu, Sigma, num_sample=1, quasimc=False):

                # x = mvn(mu, Sigma)
                factor = cholesky(Sigma, lower=True)

                if num_sample == 1:
                    if quasimc:
                        u = random_sequence_rqmc(dim=dim, n=1)
                        x = transform_uniform_to_normal(u, mu, Sigma)
                        x = np.squeeze(x)
                    else:
                        x = mu + (factor @ np.random.randn(dim, num_sample)).squeeze()
                    nabla_1 = Sigma_true_inv @ (x - mu_true)
                else:
                    if quasimc:
                        u = random_sequence_rqmc(dim=dim, n=num_sample)
                        x = transform_uniform_to_normal(u, mu, Sigma)
                        x = x.T
                    else:
                        x = mu[:, np.newaxis] + (
                            factor @ np.random.randn(dim, num_sample)
                        )
                    # print("x shape: ", x.shape)
                    nabla_1 = Sigma_true_inv @ (x - mu_true[:, np.newaxis])
                    nabla_1 = np.mean(nabla_1, axis=1)

                nabla_2 = Sigma_true_inv

                return nabla_1, nabla_2

        elif estimator == "SVRG":

            def gradient_oracle(
                mu, Sigma, get_mc_grad=False, num_sample=1, quasimc=False
            ):  # now it is variance-reduction
                # c = 0.9
                if num_sample != 1:
                    warnings.warn("num_sample is not used in SVRG estimator.")
                if quasimc:
                    warnings.warn("quasiMC not applicable for SVRG estimator.")
                # x = mvn(mu, Sigma)
                factor, lower = cho_factor(Sigma, lower=True)
                x = mu + (np.tril(factor) @ np.random.randn(dim, 1)).squeeze()

                nabla_1 = Sigma_true_inv @ (x - mu_true) - c * cho_solve(
                    (factor, lower), x - mu
                )
                nabla_2 = Sigma_true_inv
                if get_mc_grad:
                    nabla_mc = Sigma_true_inv @ (x - mu_true)
                    return nabla_1, nabla_2, nabla_mc
                return nabla_1, nabla_2

        else:
            raise NotImplemented

        if W2:
            # distance objective is the W2 distance
            def dist_objective(mu, Sigma):
                return np.log(wasserstein_dist(mu, mu_true, Sigma, Sigma_true))

        else:
            # distance objective is the KL divergence to target
            def dist_objective(mu, Sigma):
                return np.log(kl_divergence(mu, mu_true, Sigma, Sigma_true))

        def samples_sinkhorn(samples1):
            return sinkhorn_divergence(
                samples1=samples1, mu2=mu_true, Sigma2=Sigma_true
            )

        self.samples_sinkhorn = samples_sinkhorn

        def analytical_sinkhorn(mu1, Sigma1):
            return sinkhorn_divergence(
                mu1=mu1, Sigma1=Sigma1, mu2=mu_true, Sigma2=Sigma_true
            )

        self.analytical_sinkhorn = analytical_sinkhorn

        super().__init__(
            log_potential_pure, gradient_oracle, dist_objective, jax_lp, dim=dim
        )
