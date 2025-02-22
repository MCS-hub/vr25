# math
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize

# plotting/visualization
import matplotlib.pyplot as plt
from cycler import cycler
from .utils import cho_inv
import time
from .advi import ADVI
import optax
import jax


class Experiment:
    """
    Parent class for experiments.
    Implements BWGD and FBGVI and plots results.
    """

    def __init__(
        self,
        log_potential,
        gradient_oracle,
        dist_objective,
        jax_lp=None,
        dim=5,
        beta=1,
        grad_V=None,
        hess_V=None,
    ):
        """
        Initialize parameters of experiment:
        - params: dim, beta
        - log_potential
            * takes in a vector x and produces the unnormalized log potential.
        - dist_objective
            * (mu, Sigma) -> objective function to plot
            * by default, we can make this equal to EE_p V - H(p)
        - gradient_oracle
            * (mu, Sigma) -> estimate for (EE nabla V, EE nabla^2 V)
            * default:
                > draw x ~ N(mu, Sigma)
                > compute v = V(x)
                > v.backward()
                > x.grad
        - initialize Sigma_0 at isotropic Gaussian
        """

        self.log_potential = log_potential
        self.gradient_oracle = gradient_oracle
        self.dist_objective = dist_objective
        self.jax_lp = jax_lp
        self.dim = dim
        self.grad_V = grad_V
        self.hess_V = hess_V

        # initialize mean with all zeros
        self.init_mu = np.zeros(dim)

        # initialize Sigma at identity
        self.init_Sigma = np.eye(dim) / beta

    def laplace_approx(self):
        """
        Compute Laplace approximation to target distribution.
        """
        if self.hess_V is None:
            raise NotImplementedError

        x0 = np.zeros(self.dim)

        # compute MAP
        start_time = time.perf_counter()
        result = minimize(self.log_potential, x0, jac=self.grad_V, hess=self.hess_V)
        time_elapsed = time.perf_counter() - start_time
        # print("time_LA:", time_elapsed)
        mu = result.x
        # print("map", mu)
        # print("grad norm", np.linalg.norm(self.grad_V(mu)))

        # compute Hessian at map
        H = self.hess_V(mu)

        # compute inverse of Hessian to obtain Sigma
        while True:
            try:
                Sigma = cho_inv(H, self.dim)
                break
            except:
                H = H + 1e-4 * np.eye(self.dim)

        return mu, Sigma

    def advi(
        self,
        seed=1,
        lr=1e-3,
        n_iter=10000,
        batch_size=1,
        stl_estimator=True,
        diag_cov=False,
    ):
        alg = ADVI(self.dim, self.jax_lp, stl_estimator=stl_estimator)
        opt = optax.adam(learning_rate=lr)
        key = jax.random.PRNGKey(seed)
        # means, Sigmas, losses = alg.fit_with_history(key=key, opt=opt, niter=n_iter)
        # return means, Sigmas, losses
        mean, Sigma, losses = alg.fit(
            key=key, opt=opt, niter=n_iter, batch_size=batch_size
        )
        return mean, Sigma, losses

    def run_iter(self, alg, eta, mu, Sigma, num_sample, quasimc=False):
        """
        Run an iteration of the specified algorithm to update (mu, Sigma).
        Returns new (mu, Sigma).
        """

        if alg == "fbgvi":
            hat_nabla1, hat_nabla2 = self.gradient_oracle(
                mu, Sigma, num_sample=num_sample, quasimc=quasimc
            )  # hat_nabla1 is b_k, hat_nabla2 is S_k

            # compute gradient in mu
            # (update is the same between BWGD and FBGVI)
            # grad_mu = self.Sigma_true_inv @ (mu - self.mu_true)
            mu = mu - eta * hat_nabla1
            # do forward step for the energy
            M_half = np.eye(self.dim) - eta * hat_nabla2
            Sigma_half = M_half @ Sigma @ M_half

            # do backward step for the entropy
            sqrt_matrix = sqrtm(Sigma_half @ (Sigma_half + 4 * eta * np.eye(self.dim)))
            sqrt_matrix = np.real(sqrt_matrix)
            Sigma = 0.5 * (Sigma_half + 2 * eta * np.eye(self.dim) + sqrt_matrix)

            return mu, Sigma
        elif alg == "bwgd":
            hat_nabla1, hat_nabla2 = self.gradient_oracle(
                mu, Sigma
            )  # hat_nabla1 is b_k, hat_nabla2 is S_k

            # compute gradient in mu
            # (update is the same between BWGD and FBGVI)
            # grad_mu = self.Sigma_true_inv @ (mu - self.mu_true)
            mu = mu - eta * hat_nabla1
            # compute gradient in Sigma
            M = np.eye(self.dim) - eta * (hat_nabla2 - cho_inv(Sigma, self.dim))
            Sigma = M @ Sigma @ M

            return mu, Sigma

        else:
            raise NotImplementedError

    def run_experiment(
        self,
        iters_list=[10000],
        time_iter=10000,
        init_mu=None,
        init_Sigma=None,
        alg="fbgvi",
        get_var=False,
        N_var_est=5000,
        num_sample=1,
        quasimc=False,
    ):

        if init_mu is None:
            init_mu = self.init_mu
        if init_Sigma is None:
            init_Sigma = self.init_Sigma

        ax_list = list()
        prox_dists_list = list()
        time_elapsed_list = list()

        if get_var:
            var_sgvi_list_2d = list()
            var_svrgvi_list_2d = list()

        for iters in iters_list:
            eta = time_iter / iters
            prox_mu_k, prox_Sigma_k = init_mu, init_Sigma

            mu_list = [prox_mu_k]
            Sigma_list = [prox_Sigma_k]

            start_time = time.perf_counter()
            for k in range(iters):
                new_mu, new_Sigma = self.run_iter(
                    alg,
                    eta,
                    prox_mu_k,
                    prox_Sigma_k,
                    num_sample=num_sample,
                    quasimc=quasimc,
                )
                prox_mu_k, prox_Sigma_k = new_mu, new_Sigma
                mu_list.append(prox_mu_k)
                Sigma_list.append(prox_Sigma_k)
            time_elapsed = time.perf_counter() - start_time
            # print("time_other: ", time_elapsed)
            prox_dists = []
            if get_var:
                var_sgvi_list = list()
                var_svrgvi_list = list()

            for k in range(iters + 1):
                prox_mu_k = mu_list[k]
                prox_Sigma_k = Sigma_list[k]
                prox_dists.append(self.dist_objective(prox_mu_k, prox_Sigma_k))
                if get_var:
                    grad_sgvi_samples = list()
                    grad_svrgvi_samples = list()
                    # Compute the variance
                    for i in range(N_var_est):
                        nabla_1, nabla_2, nabla_mc = self.gradient_oracle(
                            prox_mu_k, prox_Sigma_k, True
                        )
                        grad_sgvi_samples.append(nabla_mc)
                        grad_svrgvi_samples.append(nabla_1)

                    grad_sgvi_samples = np.array(grad_sgvi_samples)
                    grad_svrgvi_samples = np.array(grad_svrgvi_samples)
                    grad_sgvi_mean = np.mean(grad_sgvi_samples, axis=0)
                    grad_svrgvi_mean = np.mean(grad_svrgvi_samples, axis=0)
                    var_sgvi = np.mean(
                        np.sum((grad_sgvi_samples - grad_sgvi_mean) ** 2, axis=1)
                    )
                    var_svrgvi = np.mean(
                        np.sum((grad_svrgvi_samples - grad_svrgvi_mean) ** 2, axis=1)
                    )

                    var_sgvi_list.append(var_sgvi)
                    var_svrgvi_list.append(var_svrgvi)

            ax_list.append(np.linspace(0, time_iter, iters + 1))
            prox_dists_list.append(prox_dists)
            time_elapsed_list.append(time_elapsed)

            if get_var:
                var_sgvi_list_2d.append(var_sgvi_list)
                var_svrgvi_list_2d.append(var_svrgvi_list)

        if get_var:
            return (
                ax_list,
                prox_dists_list,
                time_elapsed_list,
                var_sgvi_list_2d,
                var_svrgvi_list_2d,
            )
        else:
            return ax_list, prox_dists_list, time_elapsed_list


# plot as a seperate function


def plot_results(
    time_axes,
    alg_dists,
    ax=None,
):
    """
    Plot distances and label results for all the algorithms of interest.
    """

    # configure plot

    # plt.title(fr"{self.dist_objective_name} over iterations")
    plt.xlabel(r"time elapsed ($\eta \times \# $iters)")
    # plt.ylabel(self.dist_objective_name)

    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.set_dpi(160)

    if ax is None:
        ax = plt.gca()
    ax.set_prop_cycle(
        cycler("color", list(plt.rcParams["axes.prop_cycle"].by_key()["color"]))
        * cycler("linestyle", ["-", "--"])
    )

    for time, (alg_name, dists) in zip(time_axes, alg_dists):
        """
        Plot distances and label results for particular algorithm of interest.
        """
        dists = np.array(dists)
        ax.plot(time, dists, label=alg_name, linewidth=0.8)
    # if plot_laplace:
    #     # draw laplace approximation error
    #     laplace_mu, laplace_Sigma = self.laplace_approx()
    #     ax.axhline(
    #         self.dist_objective(laplace_mu, laplace_Sigma),
    #         label="Laplace approx"
    #     )

    plt.legend()
    plt.show()
