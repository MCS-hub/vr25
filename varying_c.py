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
    "--save-stats", type=bool, default=True, help="Save the stats (True/False)"
)

parser.add_argument(
    "--save-plot", type=bool, default=True, help="Save the plot (True/False)"
)
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--num_sample", type=int, default=1)

args = parser.parse_args()
experiment = args.experiment
iterations = args.num_iter
time_iterations = args.total_time
seed = args.seed
dim = args.dim
save_plot = args.save_plot
save_statistics = args.save_stats

if __name__ == "__main__":

    save_path = f"log/varying_c/{experiment}/dim={dim}/seed={seed}"
    if experiment == "gaussian":
        n_exp = 10
        title = r"$D_\mathsf{KL}(\mu_k \Vert \hat{\pi})$"
    else:
        raise ValueError("Invalid experiment")


    stats = {"num_iterations": iterations, "total_iteration_time": time_iterations}
    if experiment == "gaussian":
        print("Gaussian experiment...")
        # iters_list=[300], time=400
        n_exp = 10
        iters_list = [iterations] * n_exp
        alpha_choice = 5e-3

        exp_SVRG_00 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG", c=0.0
        )
        exp_SVRG_05 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG", c=0.5
        )
        exp_SVRG_08 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG", c=0.8
        )
        exp_SVRG_10 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG", c=1.0
        )
        exp_SVRG_12 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG", c=1.2
        )
        exp_SVRG_15 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG", c=1.5
        )
        exp_SVRG_20 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG", c=2.0
        )

        iter_SVRG_00, loss_SVRG_00, time_SVRG_00 = exp_SVRG_00.run_experiment(
            alg="fbgvi", iters_list=iters_list, time_iter=time_iterations
        )
        iter_SVRG_05, loss_SVRG_05, time_SVRG_05 = exp_SVRG_05.run_experiment(
            alg="fbgvi", iters_list=iters_list, time_iter=time_iterations
        )
        iter_SVRG_08, loss_SVRG_08, time_SVRG_08 = exp_SVRG_08.run_experiment(
            alg="fbgvi", iters_list=iters_list, time_iter=time_iterations
        )
        iter_SVRG_10, loss_SVRG_10, time_SVRG_10 = exp_SVRG_10.run_experiment(
            alg="fbgvi", iters_list=iters_list, time_iter=time_iterations
        )
        iter_SVRG_12, loss_SVRG_12, time_SVRG_12 = exp_SVRG_12.run_experiment(
            alg="fbgvi", iters_list=iters_list, time_iter=time_iterations
        )
        iter_SVRG_15, loss_SVRG_15, time_SVRG_15 = exp_SVRG_15.run_experiment(
            alg="fbgvi", iters_list=iters_list, time_iter=time_iterations
        )
        iter_SVRG_20, loss_SVRG_20, time_SVRG_20 = exp_SVRG_20.run_experiment(
            alg="fbgvi", iters_list=iters_list, time_iter=time_iterations
        )

        title = r"$D_\mathsf{KL}(\mu_k \Vert \hat{\pi})$"

    else:
        raise ValueError("Invalid experiment name")

    os.makedirs(save_path, exist_ok=True)
    if save_statistics:
        stats["time_iter"] = time_iterations
        stats["iters"] = iterations
        stats["execution_time_SVRG_00"] = time_SVRG_00
        stats["prox_dist_SVRG_00"] = loss_SVRG_00
        stats["execution_time_SVRG_05"] = time_SVRG_05
        stats["prox_dist_SVRG_05"] = loss_SVRG_05
        stats["execution_time_SVRG_08"] = time_SVRG_08
        stats["prox_dist_SVRG_08"] = loss_SVRG_08
        stats["execution_time_SVRG_10"] = time_SVRG_10
        stats["prox_dist_SVRG_10"] = loss_SVRG_10
        stats["execution_time_SVRG_12"] = time_SVRG_12
        stats["prox_dist_SVRG_12"] = loss_SVRG_12
        stats["execution_time_SVRG_15"] = time_SVRG_15
        stats["prox_dist_SVRG_15"] = loss_SVRG_15
        stats["execution_time_SVRG_20"] = time_SVRG_20
        stats["prox_dist_SVRG_20"] = loss_SVRG_20

        with open(f"{save_path}/stats_vary_c.json", "w") as json_file:
            json.dump(stats, json_file, indent=4)
        with open(f"{save_path}/config_vary_c.json", "w") as json_file:
            json.dump(vars(args), json_file, indent=4)


    if save_plot:

        if experiment == "gaussian" or experiment == "student_t":
            loss_SVRG_00 = np.array(loss_SVRG_00)
            loss_SVRG_05 = np.array(loss_SVRG_05)
            loss_SVRG_08 = np.array(loss_SVRG_08)
            loss_SVRG_10 = np.array(loss_SVRG_10)
            loss_SVRG_12 = np.array(loss_SVRG_12)
            loss_SVRG_15 = np.array(loss_SVRG_15)
            loss_SVRG_20 = np.array(loss_SVRG_20)

            loss_SVRG_00_e10 = 10**loss_SVRG_00
            loss_SVRG_00_e10_avg = np.mean(loss_SVRG_00_e10, axis=0)
            loss_SVRG_05_e10 = 10**loss_SVRG_05
            loss_SVRG_05_e10_avg = np.mean(loss_SVRG_05_e10, axis=0)
            loss_SVRG_08_e10 = 10**loss_SVRG_08
            loss_SVRG_08_e10_avg = np.mean(loss_SVRG_08_e10, axis=0)
            loss_SVRG_10_e10 = 10**loss_SVRG_10
            loss_SVRG_10_e10_avg = np.mean(loss_SVRG_10_e10, axis=0)
            loss_SVRG_12_e10 = 10**loss_SVRG_12
            loss_SVRG_12_e10_avg = np.mean(loss_SVRG_12_e10, axis=0)
            loss_SVRG_15_e10 = 10**loss_SVRG_15
            loss_SVRG_15_e10_avg = np.mean(loss_SVRG_15_e10, axis=0)
            loss_SVRG_20_e10 = 10**loss_SVRG_20
            loss_SVRG_20_e10_avg = np.mean(loss_SVRG_20_e10, axis=0)

            plt.figure()

            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_00_e10_avg,
                label="c=0.0",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_05_e10_avg,
                label="c=0.5",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_08_e10_avg,
                label="c=0.8",
            )

            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_10_e10_avg,
                label="c=1.0",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_12_e10_avg,
                label="c=1.2",
            )

            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_15_e10_avg,
                label="c=1.5",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_20_e10_avg,
                label="c=2.0",
            )

            plt.yscale("log")
            plt.xlabel(r"time elapsed ($\eta \times \# $iters)", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(title, fontsize=16)
            plt.legend(fontsize=16)
            plt.savefig(
                f"{save_path}/{experiment}_fig_dim={dim}_vary_c.pdf",
                format="pdf",
                bbox_inches="tight",
            )

            plt.figure()
            c_list = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
            y_value = [
                loss_SVRG_00_e10_avg[-1],
                loss_SVRG_05_e10_avg[-1],
                loss_SVRG_08_e10_avg[-1],
                loss_SVRG_10_e10_avg[-1],
                loss_SVRG_12_e10_avg[-1],
                loss_SVRG_15_e10_avg[-1],
                loss_SVRG_20_e10_avg[-1],
            ]
            plt.scatter(c_list, y_value, s=120)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.yscale("log")
            plt.xlabel("c", fontsize=16)
            plt.ylabel(title, fontsize=16)
            plt.savefig(
                f"{save_path}/{experiment}_fig_dim={dim}_vary_c_last_value.pdf",
                format="pdf",
                bbox_inches="tight",
            )

        elif experiment == "logistic":

            loss_SVRG_00 = np.array(loss_SVRG_00)
            loss_SVRG_05 = np.array(loss_SVRG_05)
            loss_SVRG_08 = np.array(loss_SVRG_08)
            loss_SVRG_10 = np.array(loss_SVRG_10)
            loss_SVRG_12 = np.array(loss_SVRG_12)
            loss_SVRG_15 = np.array(loss_SVRG_15)
            loss_SVRG_20 = np.array(loss_SVRG_20)

            loss_SVRG_00_min = np.min(loss_SVRG_00)
            loss_SVRG_05_min = np.min(loss_SVRG_05)
            loss_SVRG_08_min = np.min(loss_SVRG_08)
            loss_SVRG_10_min = np.min(loss_SVRG_10)
            loss_SVRG_12_min = np.min(loss_SVRG_12)
            loss_SVRG_15_min = np.min(loss_SVRG_15)
            loss_SVRG_20_min = np.min(loss_SVRG_20)

            loss_min = min(
                loss_SVRG_00_min,
                loss_SVRG_05_min,
                loss_SVRG_08_min,
                loss_SVRG_10_min,
                loss_SVRG_12_min,
                loss_SVRG_15_min,
                loss_SVRG_20_min,
            )

            loss_SVRG_00_avg = np.mean(loss_SVRG_00, axis=0)
            loss_SVRG_05_avg = np.mean(loss_SVRG_05, axis=0)
            loss_SVRG_08_avg = np.mean(loss_SVRG_08, axis=0)
            loss_SVRG_10_avg = np.mean(loss_SVRG_10, axis=0)
            loss_SVRG_12_avg = np.mean(loss_SVRG_12, axis=0)
            loss_SVRG_15_avg = np.mean(loss_SVRG_15, axis=0)
            loss_SVRG_20_avg = np.mean(loss_SVRG_20, axis=0)

            plt.figure()
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_00_avg - loss_min,
                label="c=0.0",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_05_avg - loss_min,
                label="c=0.5",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_08_avg - loss_min,
                label="c=0.8",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_10_avg - loss_min,
                label="c=1.0",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_12_avg - loss_min,
                label="c=1.2",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_15_avg - loss_min,
                label="c=1.5",
            )
            plt.plot(
                iter_SVRG_00[0],
                loss_SVRG_20_avg - loss_min,
                label="c=2.0",
            )

            plt.xlabel(r"time elapsed ($\eta \times \# $iters)", fontsize=16)
            plt.yscale("log")
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(title, fontsize=16)
            plt.legend(fontsize=16, loc="upper right")
            plt.savefig(
                f"{save_path}/logis_fig_dim={dim}_vary_c.pdf",
                format="pdf",
                bbox_inches="tight",
            )

            plt.figure()
            c_list = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
            y_value = [
                loss_SVRG_00_avg[-1] - loss_min,
                loss_SVRG_05_avg[-1] - loss_min,
                loss_SVRG_08_avg[-1] - loss_min,
                loss_SVRG_10_avg[-1] - loss_min,
                loss_SVRG_12_avg[-1] - loss_min,
                loss_SVRG_15_avg[-1] - loss_min,
                loss_SVRG_20_avg[-1] - loss_min,
            ]
            plt.scatter(c_list, y_value, s=120)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.yscale("log")
            plt.xlabel("c", fontsize=16)
            plt.ylabel(title, fontsize=16)
            plt.savefig(
                f"{save_path}/logis_fig_dim={dim}_vary_c_last_value.pdf",
                format="pdf",
                bbox_inches="tight",
            )
