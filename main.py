import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

from svrfbe.logreg import LogisticRegressionExperiment
from svrfbe.multgauss import GaussianExperimentEstimator
from svrfbe.student_t import StudentTExperimentEstimator
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="gaussian")

parser.add_argument("--num_iter", type=int, default=300, help="Number of iterations")
parser.add_argument("--total_time", type=float, default=300, help="Time for stepsize")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--not-save-stats",
    action="store_true",
    help="Save the stats (True/False)",
)

parser.add_argument(
    "--not-save-plot", action="store_true", help="Save the plot (True/False)"
)
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--num_sample", type=int, default=1)
parser.add_argument("--get_var", action="store_true")

args = parser.parse_args()
experiment = args.experiment
iterations = args.num_iter
time_iterations = args.total_time
seed = args.seed
dim = args.dim
save_plot = not args.not_save_plot
save_statistics = not args.not_save_stats
get_var = args.get_var

if __name__ == "__main__":
    save_path = f"log/main_exp/{experiment}/dim={dim}/seed={seed}"
    if experiment == "gaussian":
        n_exp = 10
        title = r"$D_\mathsf{KL}(\mu_k \Vert \hat{\pi})$"
    elif experiment == "gaussian_times":
        n_exp = 10
        title = r"$D_\mathsf{KL}(\mu_k \Vert \hat{\pi})$"
    elif experiment == "student_t":
        n_exp = 10
        title = r"$D_{\mathsf{KL}}(\mu_k \Vert \hat{\pi})$ (empirical)"
    elif experiment == "logistic":
        n_exp = 10
        title = r"$\mathcal{F}(\mu_k)-\mathcal{F}(\mu_{\text{best}})$ (empirical)"

    # get_var = True  # compute the variance along iterations for plotting

    stats = {"num_iterations": iterations, "total_iteration_time": time_iterations}
    if experiment == "gaussian":
        print("Gaussian experiment...")
        # iters_list=[300], time=400
        n_exp = 10
        iters_list = [iterations] * n_exp
        alpha_choice = 5e-3
        exp_SG = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        exp_SVRG = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG"
        )
        exp_BWGD = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        iter_SG, loss_SG, time_SG = exp_SG.run_experiment(
            alg="fbgvi",
            iters_list=iters_list,
            time_iter=time_iterations,
        )

        if get_var:
            (
                iter_SVRG,
                loss_SVRG,
                time_SVRG,
                var_sgvi_list_2d,
                var_svrgvi_list_2d,
            ) = exp_SVRG.run_experiment(
                alg="fbgvi",
                iters_list=iters_list,
                time_iter=time_iterations,
                get_var=True,
            )
        else:
            iter_SVRG, loss_SVRG, time_SVRG = exp_SVRG.run_experiment(
                alg="fbgvi",
                iters_list=iters_list,
                time_iter=time_iterations,
                get_var=False,
            )
        iter_BWGD, loss_BWGD, time_BWGD = exp_BWGD.run_experiment(
            alg="bwgd", iters_list=iters_list, time_iter=time_iterations
        )

        # ADVI
        loss_ADVI = []
        loss_LA = []
        lr = 1e-2
        n_iter = 5_000
        if dim in [100, 200]:
            lr = 1e-3
            n_iter = 10_000

        for i in range(n_exp):
            mu_ADVI, Sigma_ADVI, losses = exp_SG.advi(
                seed=seed + i, lr=lr, n_iter=n_iter
            )
            loss_ADVI.append(exp_SG.dist_objective(mu_ADVI, Sigma_ADVI))

        title = r"$D_\mathsf{KL}(\mu_k \Vert \hat{\pi})$"

    elif experiment == "gaussian_times":
        print("Gaussian experiment...")
        # iters_list=[300], time=400
        n_exp = 10
        iters_list = [iterations] * n_exp
        alpha_choice = 5e-3
        exp_SG = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        exp_SVRG = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG"
        )
        exp_BWGD = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        all_time_iterations = [
            time_iterations / 8,
            time_iterations / 4,
            time_iterations / 2,
            time_iterations,
        ]
        loss_SGs = []
        loss_SVRGs = []
        loss_BWGDs = []
        for time_iterations in all_time_iterations:
            print(time_iterations)
            _, loss_SG, _ = exp_SG.run_experiment(
                alg="fbgvi",
                iters_list=iters_list,
                time_iter=time_iterations,
            )
            loss_SGs.append(loss_SG)

            _, loss_SVRG, _ = exp_SVRG.run_experiment(
                alg="fbgvi",
                iters_list=iters_list,
                time_iter=time_iterations,
            )
            loss_SVRGs.append(loss_SVRG)

            _, loss_BWGD, _ = exp_BWGD.run_experiment(
                alg="bwgd", iters_list=iters_list, time_iter=time_iterations
            )
            loss_BWGDs.append(loss_BWGD)

        # ADVI
        loss_ADVI = []
        loss_LA = []
        lr = 1e-2
        n_iter = 5_000
        if dim in [100, 200]:
            lr = 1e-3
            n_iter = 10_000

        for i in range(n_exp):
            mu_ADVI, Sigma_ADVI, losses = exp_SG.advi(
                seed=seed + i, lr=lr, n_iter=n_iter
            )
            loss_ADVI.append(exp_SG.dist_objective(mu_ADVI, Sigma_ADVI))

        # title = r"$D_\mathsf{KL}(\mu_k \Vert \hat{\pi})$"

    elif experiment == "student_t":
        print("Student's t experiment...")
        n_exp = 10
        iters_list = [iterations] * n_exp
        alpha_choice = 5e-3
        exp_SG = StudentTExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        exp_BWGD = StudentTExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        exp_SVRG = StudentTExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG"
        )
        iter_SG, loss_SG, time_SG = exp_SG.run_experiment(
            alg="fbgvi", iters_list=iters_list, time_iter=time_iterations
        )
        iter_SVRG, loss_SVRG, time_SVRG = exp_SVRG.run_experiment(
            alg="fbgvi",
            iters_list=iters_list,
            time_iter=time_iterations,
            get_var=get_var,
        )
        iter_BWGD, loss_BWGD, time_BWGD = exp_BWGD.run_experiment(
            alg="bwgd", iters_list=iters_list, time_iter=time_iterations
        )

        # Laplace
        mu_LA, Sigma_LA = exp_SG.laplace_approx()
        loss_LA = exp_SG.dist_objective(mu_LA, Sigma_LA)

        # ADVI
        loss_ADVI = []
        lr = 1e-3
        n_iter = 8_000

        for i in range(n_exp):
            mu_ADVI, Sigma_ADVI, losses = exp_SG.advi(
                seed=seed + i, lr=lr, n_iter=n_iter
            )
            loss_ADVI.append(exp_SG.dist_objective(mu_ADVI, Sigma_ADVI))

        title = r"$D_{\mathsf{KL}}(\mu_k \Vert \hat{\pi})$ (empirical)"

    elif experiment == "logistic":
        print("logistic experiment...")


        n_exp = 10
        iters_list = [iterations] * n_exp

        exp_SG = LogisticRegressionExperiment(
            dim=dim,
            n=1000,
            seed=seed,
            FI=False,
            estimator="SG",
        )
        exp_BWGD = LogisticRegressionExperiment(
            dim=dim,
            n=1000,
            seed=seed,
            FI=False,
            estimator="SG",
        )
        exp_SVRG = LogisticRegressionExperiment(
            dim=dim,
            n=1000,
            seed=seed,
            FI=False,
            estimator="SVRG",
        )
        # Laplace
        mu_LA, Sigma_LA = exp_SG.laplace_approx()
        loss_LA = exp_SG.dist_objective(mu_LA, Sigma_LA)

        iter_SG, loss_SG, time_SG = exp_SG.run_experiment(
            alg="fbgvi", iters_list=iters_list, time_iter=time_iterations
        )
        iter_SVRG, loss_SVRG, time_SVRG = exp_SVRG.run_experiment(
                alg="fbgvi",
                iters_list=iters_list,
                time_iter=time_iterations,
                get_var=get_var,
                N_var_est=500,
            )
        iter_BWGD, loss_BWGD, time_BWGD = exp_BWGD.run_experiment(
            alg="bwgd", iters_list=iters_list, time_iter=time_iterations
        )

        # Laplace
        mu_LA, Sigma_LA = exp_SG.laplace_approx()
        loss_LA = exp_SG.dist_objective(mu_LA, Sigma_LA)

        # ADVI
        loss_ADVI = []
        lr = 1e-2
        n_iter = 3_000

        for i in range(n_exp):
            mu_ADVI, Sigma_ADVI, losses = exp_SG.advi(
                seed=seed + i, lr=lr, n_iter=n_iter
            )
            loss_ADVI.append(exp_SG.dist_objective(mu_ADVI, Sigma_ADVI))

        title = r"$\mathcal{F}(\mu_k)-\mathcal{F}(\mu_{\mathrm{best}})$ (empirical)"

    else:
        raise ValueError("Invalid experiment name")

    os.makedirs(save_path, exist_ok=True)
    if save_statistics:
        if experiment != "gaussian_times":
            stats["time_iter"] = time_iterations
            stats["iters"] = iterations
            stats["execution_time_SG"] = time_SG
            stats["prox_dist_SG"] = loss_SG
            stats["execution_time_SVRG"] = time_SVRG
            stats["prox_dist_SVRG"] = loss_SVRG
            stats["execution_time_BWGD"] = time_BWGD
            stats["prox_dist_BWGD"] = loss_BWGD
            stats["prox_dist_LA"] = loss_LA
            stats["prox_dist_ADVI"] = loss_ADVI
            if get_var:
                stats["var_sgvi_list_2d"] = var_sgvi_list_2d
                stats["var_svrgvi_list_2d"] = var_svrgvi_list_2d
        else:
            stats["all_time_iters"] = all_time_iterations
            stats["iters"] = iterations
            stats["prox_dist_BWGDs"] = loss_BWGDs
            stats["prox_dist_SGs"] = loss_SGs
            stats["prox_dist_SVRGs"] = loss_SVRGs
            stats["prox_dist_AVDI"] = loss_ADVI
        with open(f"{save_path}/stats.json", "w") as json_file:
            json.dump(stats, json_file, indent=4)
        with open(f"{save_path}/config.json", "w") as json_file:
            json.dump(vars(args), json_file, indent=4)


    if save_plot:
        advi_color = "goldenrod"
        advi_label = "EVI"

        if experiment == "gaussian" or experiment == "student_t":
            loss_BWGD = np.array(loss_BWGD)
            loss_SG = np.array(loss_SG)
            loss_SVRG = np.array(loss_SVRG)
            loss_ADVI = np.array(loss_ADVI)

            loss_BWGD_e10 = 10**loss_BWGD
            loss_SG_e10 = 10**loss_SG
            loss_SVRG_e10 = 10**loss_SVRG
            loss_ADVI_e10 = 10**loss_ADVI
            if experiment != "gaussian":
                loss_LA_e10 = 10**loss_LA
                loss_LA_e10 = np.array([loss_LA_e10] * len(iter_SG[0]))

            loss_BWGD_e10_avg = np.mean(loss_BWGD_e10, axis=0)
            loss_SG_e10_avg = np.mean(loss_SG_e10, axis=0)
            loss_SVRG_e10_avg = np.mean(loss_SVRG_e10, axis=0)
            loss_ADVI_e10_avg = np.mean(loss_ADVI_e10)

            loss_ADVI_e10 = np.repeat(loss_ADVI_e10[:, None], len(iter_SG[0]), axis=1)
            loss_ADVI_e10_avg = np.repeat(loss_ADVI_e10_avg, len(iter_SG[0]))
            
            plt.figure()
            plt.plot(
                iter_BWGD[0],
                loss_BWGD_e10_avg,
                label="BWGD",
                color="royalblue",
            )
            plt.plot(iter_SG[0], loss_SG_e10_avg, label="SGVI", color="crimson")
            plt.plot(
                iter_SVRG[0],
                loss_SVRG_e10_avg,
                label="SVRGVI (ours)",
                color="black",
            )
            if experiment != "gaussian":
                plt.plot(
                    iter_SG[0],
                    loss_LA_e10,
                    color="darkgreen",
                    linestyle="--",
                    label="Laplace",
                )
            plt.plot(
                iter_SG[0],
                loss_ADVI_e10_avg,
                color=advi_color,
                linestyle="--",
                label=advi_label,
            )

            for i in range(n_exp):
                plt.plot(
                    iter_BWGD[0],
                    loss_BWGD_e10[i, :],
                    color="royalblue",
                    alpha=0.1,
                )
                plt.plot(
                    iter_SG[0],
                    loss_SG_e10[i, :],
                    color="crimson",
                    alpha=0.1,
                )
                plt.plot(
                    iter_SVRG[0],
                    loss_SVRG_e10[i, :],
                    color="black",
                    alpha=0.1,
                )
                plt.plot(
                    iter_SG[0],
                    loss_ADVI_e10[i, :],
                    color=advi_color,
                    alpha=0.05,
                )

            plt.yscale("log")
            plt.xlabel(r"time elapsed ($\eta \times \# $iters)", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(title, fontsize=16)
            plt.legend(fontsize=16)
            plt.savefig(
                f"{save_path}/{experiment}_fig_dim={dim}.pdf",
                format="pdf",
                bbox_inches="tight",
            )

            if get_var:
                plt.figure()
                # plot variance
                var_sgvi_list_2d = np.array(var_sgvi_list_2d)
                var_svrgvi_list_2d = np.array(var_svrgvi_list_2d)

                var_sgvi_mean = np.mean(var_sgvi_list_2d, axis=0)
                var_svrgvi_mean = np.mean(var_svrgvi_list_2d, axis=0)

                # plot mean curve
                plt.plot(
                    iter_SVRG[0],
                    var_sgvi_mean,
                    label="Monte Carlo estimator",
                    color="blue",
                )
                plt.plot(
                    iter_SVRG[0], var_svrgvi_mean, label="Our estimator", color="black"
                )
                plt.yscale("log")

                # plot background curves
                for pos in range(n_exp):
                    plt.plot(
                        iter_SVRG[0], var_sgvi_list_2d[pos], color="blue", alpha=0.2
                    )
                    plt.plot(
                        iter_SVRG[0], var_svrgvi_list_2d[pos], color="black", alpha=0.2
                    )
                    plt.yscale("log")

                plt.ylabel("Variance", fontsize=16)
                plt.xlabel(r"time elapsed ($\eta \times \# $iters)", fontsize=16)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.legend(fontsize=16)
                plt.savefig(
                    f"{save_path}/{experiment}_var_fig_dim={dim}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
        elif experiment == "gaussian_times":

            xs = np.arange(iterations + 1)

            plt.figure()

            linestyles = [":", "-.", "--", "-"]
            assert len(all_time_iterations) == len(linestyles)

            step_sizes = [
                time_iterations / iterations for time_iterations in all_time_iterations
            ]

            final_loss_BWGDs = []
            final_loss_SGs = []
            final_loss_SVRGs = []

            for index, step_size in enumerate(step_sizes):

                loss_BWGD = loss_BWGDs[index]
                loss_SG = loss_SGs[index]
                loss_SVRG = loss_SVRGs[index]

                loss_BWGD = np.array(loss_BWGD)
                loss_SG = np.array(loss_SG)
                loss_SVRG = np.array(loss_SVRG)

                loss_BWGD_e10 = 10**loss_BWGD
                loss_SG_e10 = 10**loss_SG
                loss_SVRG_e10 = 10**loss_SVRG

                loss_BWGD_e10_avg = np.mean(loss_BWGD_e10, axis=0)
                loss_SG_e10_avg = np.mean(loss_SG_e10, axis=0)
                loss_SVRG_e10_avg = np.mean(loss_SVRG_e10, axis=0)

                final_loss_BWGDs.append(loss_BWGD_e10_avg[-1])
                final_loss_SGs.append(loss_SG_e10_avg[-1])
                final_loss_SVRGs.append(loss_SVRG_e10_avg[-1])

                if index == (len(all_time_iterations) - 1):
                    plt.plot(
                        xs,
                        loss_BWGD_e10_avg,
                        linestyle=linestyles[index],
                        label="BWGD",
                        color="royalblue",
                    )
                    plt.plot(
                        xs,
                        loss_SG_e10_avg,
                        linestyle=linestyles[index],
                        label="SGVI",
                        color="crimson",
                    )
                    plt.plot(
                        xs,
                        loss_SVRG_e10_avg,
                        linestyle=linestyles[index],
                        label="SVRGVI (ours)",
                        color="black",
                    )
                else:
                    # no label
                    plt.plot(
                        xs,
                        loss_BWGD_e10_avg,
                        linestyle=linestyles[index],
                        color="royalblue",
                    )
                    plt.plot(
                        xs,
                        loss_SG_e10_avg,
                        linestyle=linestyles[index],
                        color="crimson",
                    )
                    plt.plot(
                        xs,
                        loss_SVRG_e10_avg,
                        linestyle=linestyles[index],
                        color="black",
                    )

            loss_ADVI = np.array(loss_ADVI)
            loss_ADVI_e10 = 10**loss_ADVI
            loss_ADVI_e10_avg = np.mean(loss_ADVI_e10)
            loss_ADVI_e10 = np.repeat(loss_ADVI_e10[:, None], len(xs), axis=1)
            loss_ADVI_e10_avg = np.repeat(loss_ADVI_e10_avg, len(xs))

            plt.plot(
                xs,
                loss_ADVI_e10_avg,
                color=advi_color,
                linestyle="--",
                label=advi_label,
            )
            for i in range(n_exp):
                plt.plot(
                    xs,
                    loss_ADVI_e10[i, :],
                    color=advi_color,
                    alpha=0.05,
                )

            plt.yscale("log")
            plt.xlabel(r"number of iterations", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(title, fontsize=16)
            plt.legend(fontsize=16)
            plt.savefig(
                f"{save_path}/{experiment}_fig_dim={dim}.pdf",
                format="pdf",
                bbox_inches="tight",
            )

            plt.figure()
            plt.plot(
                step_sizes,
                final_loss_BWGDs,
                label="BWGD",
                color="royalblue",
            )
            plt.plot(
                step_sizes,
                final_loss_SGs,
                label="SGVI",
                color="crimson",
            )
            plt.plot(
                step_sizes,
                final_loss_SVRGs,
                label="SVRGVI (ours)",
                color="black",
            )
            plt.plot(
                step_sizes,
                [loss_ADVI_e10_avg[-1] for _ in step_sizes],
                color=advi_color,
                linestyle="--",
                label=advi_label,
            )
            plt.yscale("log")
            plt.xlabel(r"step sizes", fontsize=16)
            plt.xscale("log")
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(title, fontsize=16)
            plt.legend(fontsize=16)
            plt.savefig(
                f"{save_path}/{experiment}_fig_dim={dim}_step_sizes.pdf",
                format="pdf",
                bbox_inches="tight",
            )
        elif experiment == "logistic":
            loss_BWGD = np.array(loss_BWGD)
            loss_SG = np.array(loss_SG)
            loss_SVRG = np.array(loss_SVRG)
            loss_ADVI = np.array(loss_ADVI)

            loss_BWGD_min = np.min(loss_BWGD)
            loss_SG_min = np.min(loss_SG)
            loss_SVRG_min = np.min(loss_SVRG)
            loss_ADVI_min = np.min(loss_ADVI)
            loss_min = min(
                loss_BWGD_min, loss_SG_min, loss_SVRG_min, loss_LA, loss_ADVI_min
            )

            loss_BWGD_avg = np.mean(loss_BWGD, axis=0)
            loss_SG_avg = np.mean(loss_SG, axis=0)
            loss_SVRG_avg = np.mean(loss_SVRG, axis=0)

            loss_ADVI_avg = np.repeat(np.mean(loss_ADVI, axis=0), len(iter_SG[0]))
            loss_ADVI = np.repeat(loss_ADVI[:, None], len(iter_SG[0]), axis=1)

            loss_LA = np.array([loss_LA] * len(iter_SG[0]))
            plt.figure()
            plt.plot(
                iter_BWGD[0],
                loss_BWGD_avg - loss_min,
                label="BWGD",
                color="royalblue",
            )
            plt.plot(iter_SG[0], loss_SG_avg - loss_min, label="SGVI", color="crimson")
            plt.plot(
                iter_SVRG[0],
                loss_SVRG_avg - loss_min,
                label="SVRGVI (ours)",
                color="black",
            )
            plt.plot(
                iter_SG[0],
                loss_LA - loss_min,
                color="darkgreen",
                linestyle="--",
                label="Laplace",
            )
            plt.plot(
                iter_SG[0],
                loss_ADVI_avg - loss_min,
                color=advi_color,
                linestyle="--",
                label=advi_label,
            )
            for i in range(n_exp):
                plt.plot(
                    iter_BWGD[0],
                    loss_BWGD[i, :] - loss_min,
                    color="royalblue",
                    alpha=0.05,
                )
                plt.plot(
                    iter_SG[0],
                    loss_SG[i, :] - loss_min,
                    color="crimson",
                    alpha=0.05,
                )
                plt.plot(
                    iter_SVRG[0],
                    loss_SVRG[i, :] - loss_min,
                    color="black",
                    alpha=0.05,
                )
                plt.plot(
                    iter_SVRG[0],
                    loss_ADVI[i, :] - loss_min,
                    color=advi_color,
                    alpha=0.05,
                )
            plt.xlabel(r"time elapsed ($\eta \times \# $iters)", fontsize=16)
            plt.yscale("log")
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(title, fontsize=16)
            plt.legend(fontsize=16, loc="upper right")
            plt.savefig(
                f"{save_path}/logis_fig_dim={dim}.pdf",
                format="pdf",
                bbox_inches="tight",
            )
