# math
import numpy as np
import jax.numpy as jnp
from numpy.random import multivariate_normal as mvn
from scipy.stats import bernoulli
from scipy.linalg import cholesky, cho_factor, cho_solve
from scipy.special import expit  # Sigmoid function
from scipy.linalg import det
import warnings


# python utils
from functools import partial

from .experiment import Experiment
from .utils import softmax, H


class LogisticRegressionExperiment(Experiment):
    """
    Bayesian logistic regression posterior as target.
    """

    def __init__(
        self,
        dim=5,
        n=20,
        TOTAL_SAMPLES=1000,
        seed=0,
        estimator="SG",
        FI=False,
        do_c_opt=False,
        c=0.9,
    ):
        """
        Parameters:
        - dim: dimension of theta
        - n: number of samples from generative model
        - TOTAL_SAMPLES:
            total number of samples used to approximate KL divergence
        """
        """
        Define the log potental, gradient, Hessian.
        """
        self.c = c
        if seed is not None:
            np.random.seed(seed)

        def log_potential(theta, Y, X):
            """
            Compute log potential.
            V(theta) = sum_{i=1}^n
                (-Y_i <theta, X_i> + ln(1 + exp <theta, X_i>)).
            """
            first = np.log(1 + np.exp(np.dot(X, theta))).sum(axis=0)
            second = np.dot(Y, np.dot(X, theta))
            V = first - second
            return V

        def grad_V(theta, Y, X):
            """
            Compute gradient of log potential.
            nabla V(theta) = -sum_{i=1}^n
                (Y_i - softmax(<theta, X_i>)) X_i.
            """
            first = Y - softmax(np.dot(X, theta)).T
            second = -np.dot(first, X)

            nablaV = second.T

            return nablaV

        def hess_V(theta, Y, X):
            """
            Compute the Hessian of the log potential.
            nabla^2 V(theta) = sum_{i=1}^n
                (exp <theta, X_i>) / (1 + exp <theta, X_i>)^2 X_i X_i^T
            """
            first = softmax(np.dot(X, theta)) / (1 + np.exp(np.dot(X, theta)))
            second = X.T @ np.diag(first) @ X

            nabla2V = second
            return nabla2V

        """
        Data generation
        """
        # generate true parameter theta

        true_theta = mvn(np.zeros(dim), np.eye(dim))

        # generate random data matrix X
        X = mvn(np.zeros(dim), np.eye(dim), size=(n))

        maxeig = np.linalg.eig(X.T @ X)[0][0]
        X = X / np.sqrt(maxeig)

        # from generative model, draw observations Y
        probs = softmax(X @ true_theta)
        Y = bernoulli.rvs(probs)


        # Define log_potential as a pure function of theta
        log_potential_pure = partial(log_potential, Y=Y, X=X)
        grad_pure = partial(grad_V, Y=Y, X=X)
        hess_pure = partial(hess_V, Y=Y, X=X)

        jax_X = jnp.array(X)
        jax_Y = jnp.array(Y)

        def jax_lp(theta):
            theta = theta.T
            first = jnp.log(1 + jnp.exp(jnp.dot(jax_X, theta))).sum(axis=0)
            second = jnp.dot(jax_Y, jnp.dot(jax_X, theta))
            return second - first

        if estimator == "SG":

            def gradient_oracle(mu, Sigma, num_sample=1, quasimc=False):
                if num_sample != 1:
                    warnings.warn(
                        "num_sample is not used in the Bayesian LR experiment."
                    )
                if quasimc:
                    warnings.warn(
                        "quasimc is not used in the Bayesian LR experiment."
                    )
                factor = cholesky(Sigma, lower=True)
                theta = mu + (factor @ np.random.randn(dim, 1)).squeeze()
                return grad_pure(theta), hess_pure(theta)

        elif estimator == "SVRG":

            def gradient_oracle(mu, Sigma, get_mc_grad=False, num_sample=1, quasimc=False):
                if num_sample != 1:
                    warnings.warn(
                        "num_sample is not used in the Bayesian LR experiment."
                    )
                if quasimc:
                    warnings.warn(
                        "quasimc is not used in the Bayesian LR experiment."
                    )
                factor, lower = cho_factor(Sigma, lower=True)
                theta = mu + (np.tril(factor) @ np.random.randn(dim, 1)).squeeze()
                nabla_2 = hess_pure(theta)
                if do_c_opt:
                    # Compute optimal c
                    trace_inv_Sigma = np.sum(1.0 / (np.diag(factor) ** 2))
                    c = np.trace(nabla_2) / trace_inv_Sigma
                else:
                    c = self.c

                nabla_1 = grad_pure(theta) - c * cho_solve((factor, lower), theta - mu)
                if get_mc_grad:
                    nabla_mc = grad_pure(theta)
                    return nabla_1, nabla_2, nabla_mc
                return nabla_1, nabla_2

        if FI:
            # use empirical BW fisher info as distance objective.
            def dist_objective(mu, Sigma):
                """
                Compute the empirical log Fisher information as a distance objective.
                """
                Sigma_inv = np.linalg.inv(Sigma)

                # draw random samples from N(mu, Sigma) to approximate EE||nabla_BW F||^2.
                x = mvn(mu, Sigma, size=TOTAL_SAMPLES).T

                # grad_log = grad_pure(x) - np.dot(Sigma_inv, (x.T - mu).T)
                # assert grad_log.shape
                # return np.log((grad_log ** 2).sum(axis=0).mean())

                grad_stack = grad_pure(x)
                # should be (TOTAL_SAMPLES, dim)
                expected_grad = grad_stack.mean(axis=0)

                # compute expected hessian by integration by parts.
                expected_hess = 1 / 2 * (np.dot(grad_stack, (x.T - mu))) / TOTAL_SAMPLES
                expected_hess = expected_hess @ Sigma_inv
                # enforce symmetric
                expected_hess = expected_hess + expected_hess.T

                bw_grad = (
                    expected_grad
                    + np.dot(x.T - mu, expected_hess).T
                    - np.dot(x.T - mu, Sigma_inv).T
                )
                return np.log((bw_grad**2).sum(axis=0).mean())

        else:
            # use empirical KL divergence to unnormalized target.
            def dist_objective(mu, Sigma):
                """
                Compute the empirical KL divergence
                to the unnormalized target as a distance objective.
                """
                # draw random samples from N(mu, Sigma) to approximate F(p_k).
                theta = mvn(mu, Sigma, size=TOTAL_SAMPLES).T
                # compute empirical expectation of V
                log_potentials = log_potential_pure(theta)
                assert log_potentials.shape == (TOTAL_SAMPLES,)
                EE_V = log_potentials.mean()
                return EE_V + H(mu, Sigma)

        super().__init__(
            log_potential_pure,
            gradient_oracle,
            dist_objective,
            jax_lp,
            dim=dim,
            grad_V=grad_pure,
            hess_V=hess_pure,
        )
