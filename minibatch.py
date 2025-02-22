import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import time

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


args = parser.parse_args()
experiment = args.experiment
iterations = args.num_iter
time_iterations = args.total_time
seed = args.seed
dim = args.dim
save_plot = args.save_plot
save_statistics = args.save_stats


if __name__ == "__main__":

    save_path = f"log/minibatch/{experiment}/dim={dim}/seed={seed}"
    if experiment == "gaussian":
        n_exp = 10
        title = r"$D_\mathsf{KL}(\mu_k \Vert \hat{\pi})$"


    stats = {"num_iterations": iterations, "total_iteration_time": time_iterations}
    if experiment == "gaussian":
        print("Gaussian experiment...")
        # iters_list=[300], time=400
        n_exp = 10
        iters_list = [iterations] * n_exp
        alpha_choice = 5e-3
        exp_SG_1 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        exp_SG_10 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        exp_SG_100 = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )

        exp_SG_10_q = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        exp_SG_50_q = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )
        exp_SG_100_q = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SG"
        )

        exp_SVRG = GaussianExperimentEstimator(
            dim=dim, seed=seed, alpha=alpha_choice, estimator="SVRG"
        )

        iter_SG_1, loss_SG_1, time_SG_1 = exp_SG_1.run_experiment(
            alg="fbgvi",
            iters_list=iters_list,
            time_iter=time_iterations,
            num_sample=1,
        )
        iter_SG_10, loss_SG_10, time_SG_10 = exp_SG_10.run_experiment(
            alg="fbgvi",
            iters_list=iters_list,
            time_iter=time_iterations,
            num_sample=10,
        )
        iter_SG_100, loss_SG_100, time_SG_100 = exp_SG_100.run_experiment(
            alg="fbgvi",
            iters_list=iters_list,
            time_iter=time_iterations,
            num_sample=100,
        )

        iter_SG_10_q, loss_SG_10_q, time_SG_10_q = exp_SG_10_q.run_experiment(
            alg="fbgvi",
            iters_list=iters_list,
            time_iter=time_iterations,
            num_sample=10,
            quasimc=True,
        )
        iter_SG_50_q, loss_SG_50_q, time_SG_50_q = exp_SG_50_q.run_experiment(
            alg="fbgvi",
            iters_list=iters_list,
            time_iter=time_iterations,
            num_sample=50,
            quasimc=True,
        )
        iter_SG_100_q, loss_SG_100_q, time_SG_100_q = exp_SG_100_q.run_experiment(
            alg="fbgvi",
            iters_list=iters_list,
            time_iter=time_iterations,
            num_sample=100,
            quasimc=True,
        )

        iter_SVRG, loss_SVRG, time_SVRG = exp_SVRG.run_experiment(
            alg="fbgvi",
            iters_list=iters_list,
            time_iter=time_iterations,
        )

        title = r"$D_\mathsf{KL}(\mu_k \Vert \hat{\pi})$"

    else:
        raise ValueError("Invalid experiment name")

    os.makedirs(save_path, exist_ok=True)
    if save_statistics:
        stats["time_iter"] = time_iterations
        stats["iters"] = iterations
        stats["execution_time_SG_1"] = time_SG_1
        stats["prox_dist_SG_1"] = loss_SG_1
        stats["execution_time_SG_10"] = time_SG_10
        stats["prox_dist_SG_10"] = loss_SG_10
        stats["execution_time_SG_100"] = time_SG_100
        stats["prox_dist_SG_100"] = loss_SG_100
        stats["execution_time_SG_10_q"] = time_SG_10_q
        stats["prox_dist_SG_10_q"] = loss_SG_10_q
        stats["execution_time_SG_50_q"] = time_SG_50_q
        stats["prox_dist_SG_50_q"] = loss_SG_50_q
        stats["execution_time_SG_100_q"] = time_SG_100_q
        stats["prox_dist_SG_100_q"] = loss_SG_100_q
        stats["execution_time_SVRG"] = time_SVRG
        stats["prox_dist_SVRG"] = loss_SVRG

        with open(f"{save_path}/stats.json", "w") as json_file:
            json.dump(stats, json_file, indent=4)
        with open(f"{save_path}/config.json", "w") as json_file:
            json.dump(vars(args), json_file, indent=4)


    if save_plot:

        loss_SG_1 = np.array(loss_SG_1)
        loss_SG_10 = np.array(loss_SG_10)
        loss_SG_100 = np.array(loss_SG_100)
        loss_SG_10_q = np.array(loss_SG_10_q)
        loss_SG_50_q = np.array(loss_SG_50_q)
        loss_SG_100_q = np.array(loss_SG_100_q)
        loss_SVRG = np.array(loss_SVRG)

        loss_SG_1_e10 = 10**loss_SG_1
        loss_SG_10_e10 = 10**loss_SG_10
        loss_SG_100_e10 = 10**loss_SG_100
        loss_SG_10_q_e10 = 10**loss_SG_10_q
        loss_SG_50_q_e10 = 10**loss_SG_50_q
        loss_SG_100_q_e10 = 10**loss_SG_100_q
        loss_SVRG_e10 = 10**loss_SVRG

        loss_SG_1_e10_avg = np.mean(loss_SG_1_e10, axis=0)
        loss_SG_10_e10_avg = np.mean(loss_SG_10_e10, axis=0)
        loss_SG_100_e10_avg = np.mean(loss_SG_100_e10, axis=0)
        loss_SG_10_q_e10_avg = np.mean(loss_SG_10_q_e10, axis=0)
        loss_SG_50_q_e10_avg = np.mean(loss_SG_50_q_e10, axis=0)
        loss_SG_100_q_e10_avg = np.mean(loss_SG_100_q_e10, axis=0)
        loss_SVRG_e10_avg = np.mean(loss_SVRG_e10, axis=0)

        # plotMC
        plt.figure()

        plt.plot(
            iter_SG_1[0], loss_SG_1_e10_avg, label="SGVI, 1 MC sample", color="crimson"
        )
        plt.plot(
            iter_SG_1[0],
            loss_SG_10_e10_avg,
            label="SGVI, 10 MC samples",
            color="darkviolet",
        )
        plt.plot(
            iter_SG_1[0],
            loss_SG_100_e10_avg,
            label="SGVI, 100 MC samples",
            color="green",
        )
        plt.plot(
            iter_SG_1[0],
            loss_SVRG_e10_avg,
            label="SVRGVI, 1 MC sample",
            color="black",
        )

        plt.yscale("log")
        plt.xlabel(r"time elapsed ($\eta \times \# $iters)", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(
            f"{save_path}/{experiment}_fig_dim={dim}_varyingsample.pdf",
            format="pdf",
            bbox_inches="tight",
        )

        # plot qMC

        plt.figure()

        plt.plot(
            iter_SG_1[0],
            loss_SG_10_q_e10_avg,
            label="SGVI, 10 qMC samples",
            color="crimson",
        )
        plt.plot(
            iter_SG_1[0],
            loss_SG_50_q_e10_avg,
            label="SGVI, 50 qMC samples",
            color="darkviolet",
        )
        plt.plot(
            iter_SG_1[0],
            loss_SG_100_q_e10_avg,
            label="SGVI, 100 qMC samples",
            color="green",
        )

        plt.plot(
            iter_SG_1[0],
            loss_SVRG_e10_avg,
            label="SVRGVI, 1 MC sample",
            color="black",
        )

        plt.yscale("log")
        plt.xlabel(r"time elapsed ($\eta \times \# $iters)", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(
            f"{save_path}/{experiment}_fig_dim={dim}_varyingsample_q.pdf",
            format="pdf",
            bbox_inches="tight",
        )
