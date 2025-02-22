# math
import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.stats import uniform, multivariate_t
from scipy.linalg import cholesky, cho_factor, cho_solve
import jax.numpy as jnp
import warnings

# python utils
from functools import partial

from .experiment import Experiment
from .utils import gen_matrix_with_eigs, cho_inv, kl_divergence_given_target, H


class StudentTExperimentEstimator(Experiment):
    """
    Run convergence experiment for different algorithms on randomly generated target.
    """

    def __init__(
        self,
        dim=5,
        alpha=0.01,
        beta=1,
        nu=4,
        seed=0,
        TOTAL_SAMPLES=1000,
        estimator="G",
        do_c_opt=False,
        c=0.9,
    ):

        self.c = c
        if seed is not None:
            np.random.seed(seed)

        def log_potential(x, mu, Sigma_inv, nu):
            d = mu.shape[0]
            if np.ndim(x) == 1:
                assert x.shape[0] == mu.shape[0]
                return (
                    0.5
                    * (nu + d)
                    * np.log(
                        1 + 1 / nu * np.dot((x - mu).T, np.dot(Sigma_inv, (x - mu)))
                    )
                )
            else:
                assert x.shape[1] == mu.shape[0]
                d = mu.shape[0]
                diff = x - mu  # (n, d)
                quad_form = np.einsum("ni,ij,nj->n", diff, Sigma_inv, diff)  # (n,)
                log_potentials = (
                    0.5 * (nu + d) * np.log(1 + (1 / nu) * quad_form)
                )  # (n,)
                return log_potentials

        def grad_V(x, mu, Sigma_inv, nu):
            d = x.shape[0]
            first = np.dot(Sigma_inv, (x - mu))
            second = np.dot(x - mu, first)
            return (nu + d) / (nu + second) * first

        def hess_V(x, mu, Sigma_inv, nu):
            d = x.shape[0]
            first = np.dot(Sigma_inv, (x - mu))
            second = np.dot(x - mu, first)
            return (nu + d) / (nu + second) * Sigma_inv - 2 * (nu + d) / (
                nu + second
            ) ** 2 * np.outer(first, first)

        # randomly initialize mu, Sigma
        mu_true = uniform.rvs(size=(dim,))
        Sigma_true = gen_matrix_with_eigs(np.geomspace(1 / beta, 1 / alpha, dim))
        Sigma_true_inv = cho_inv(Sigma_true, dim)

        log_potential_pure = partial(
            log_potential, mu=mu_true, Sigma_inv=Sigma_true_inv, nu=nu
        )
        grad_pure = partial(grad_V, mu=mu_true, Sigma_inv=Sigma_true_inv, nu=nu)
        hess_pure = partial(hess_V, mu=mu_true, Sigma_inv=Sigma_true_inv, nu=nu)
        target_log_pdf = multivariate_t(loc=mu_true, shape=Sigma_true, df=nu).logpdf

        jax_mu_true = jnp.array(mu_true)
        jax_Sigma_true_inv = jnp.array(Sigma_true_inv)

        def jax_lp(x):
            if jnp.ndim(x) == 1:
                d = jax_mu_true.shape[0]
                return (
                    -0.5
                    * (nu + d)
                    * jnp.log(
                        1
                        + 1
                        / nu
                        * jnp.dot(
                            (x - jax_mu_true),
                            jnp.dot(jax_Sigma_true_inv, (x - jax_mu_true)),
                        )
                    )
                )
            else:
                assert x.shape[1] == jax_mu_true.shape[0]
                d = jax_mu_true.shape[0]
                diff = x - jax_mu_true  # (n, d)
                quad_form = jnp.einsum(
                    "ni,ij,nj->n", diff, jax_Sigma_true_inv, diff
                )  # (n,)
                log_potentials = (
                    0.5 * (nu + d) * jnp.log(1 + (1 / nu) * quad_form)
                )  # (n,)
                return -log_potentials

        if estimator == "SG":

            def gradient_oracle(mu, Sigma, num_sample=1, quasimc=False):
                if num_sample != 1:
                    warnings.warn("num_sample is not used in the student-t experiment.")
                if quasimc:
                    warnings.warn("quasimc is not used in the student-t experiment.")
                # x = mvn(mu, Sigma)
                factor = cholesky(Sigma, lower=True)
                x = mu + (factor @ np.random.randn(dim, 1)).squeeze()
                return grad_pure(x), hess_pure(x)

        elif estimator == "SVRG":

            def gradient_oracle(mu, Sigma, get_mc_grad=False, num_sample=1, quasimc=False):
                if num_sample != 1:
                    warnings.warn("num_sample is not used in the student-t experiment.")
                if quasimc:
                    warnings.warn("quasimc is not used in the student-t experiment.")
                factor, lower = cho_factor(Sigma, lower=True)
                x = mu + (np.tril(factor) @ np.random.randn(dim, 1)).squeeze()
                nabla_2 = hess_pure(x)
                if do_c_opt:
                    # Compute optimal c
                    trace_inv_Sigma = np.sum(1.0 / (np.diag(factor) ** 2))
                    c = np.trace(nabla_2) / trace_inv_Sigma
                else:
                    c = self.c

                nabla_1 = grad_pure(x) - c * cho_solve((factor, lower), x - mu)
                if get_mc_grad:
                    nabla_mc = grad_pure(x)
                    return nabla_1, nabla_2, nabla_mc
                return nabla_1, nabla_2

        else:
            raise NotImplemented

        # distance objective is the empirical KL divergence to target
        # since we use normalized target, it is greater than 0.
        def dist_objective(mu, Sigma):
            kl_div = kl_divergence_given_target(
                mu, Sigma, target_log_pdf, num_samples=TOTAL_SAMPLES
            )
            return np.log(kl_div)

        super().__init__(
            log_potential_pure,
            gradient_oracle,
            dist_objective,
            jax_lp,
            dim=dim,
            grad_V=grad_pure,
            hess_V=hess_pure,
        )
