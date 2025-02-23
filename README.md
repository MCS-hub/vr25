

# STOCHASTIC VARIANCE-REDUCED GAUSSIAN VARIATIONAL INFERENCE ON THE BURES-WASSERSTEIN MANIFOLD

This repository contains the source code of the paper:

>Hoang Phuc Hau Luu, Hanlin Yu, Bernardo Williams, Marcelo Hartmann, Arto Klami. Stochastic variance-reduced Gaussian variational inference on the Bures-Wasserstein manifold. _International Conference on Learning Representations 2025._


We leverage the code from https://github.com/mzydiao/FBGVI/blob/main/FBGVI-Experiments.ipynb and Buchholz, Alexander, Florian Wenzel, and Stephan Mandt. "Quasi-monte carlo variational inference." International Conference on Machine Learning. PMLR, 2018.

Requirements:

```bash
pip install -r requirements.txt
```

Moreover, we also need to install R and its randtoolbox package.


The experiments in the paper can be run from terminal with the following commands

#### Gaussian
```bash
python main.py --experiment gaussian --seed 42 --num_iter 300 --total_time 300 --dim 10
python main.py --experiment gaussian --seed 42 --num_iter 300 --total_time 300 --dim 50
python main.py --experiment gaussian --seed 42 --num_iter 300 --total_time 300 --dim 200
```

#### Student-t
```bash
python main.py --experiment student_t --seed 42 --num_iter 300 --total_time 300 --dim 200
```

#### Bayesian Logistic Regression
```bash
python main.py --experiment logistic --seed 42 --num_iter 300 --total_time 300 --dim 200
```

#### Varying c
```bash
python varying_c.py --experiment gaussian --seed 42 --num_iter 300 --total_time 300 --dim 100
```

#### Effect of step sizes
```bash
python main.py --experiment gaussian_times --seed 42 --num_iter 300 --total_time 300 --dim 100
```

#### Minibatch
```bash
python minibatch.py --experiment gaussian --seed 42 --num_iter 300 --total_time 300 --dim 50
```
