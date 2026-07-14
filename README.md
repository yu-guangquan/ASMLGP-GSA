# ASMLGP-GSA

Adaptive-Sampling-Driven Meta-Learned Gaussian Processes for Global Sensitivity Analysis

## Overview

This repository provides the research code for an adaptive-sampling-driven meta-learned Gaussian process framework for uncertainty-aware response prediction and global sensitivity analysis of partially updated nonlinear structural systems.

The framework combines:

- Gaussian process surrogate modeling for expensive nonlinear simulations;
- transfer of information from structurally related source tasks;
- cross-validation–Voronoi adaptive sampling;
- a leave-one-out-error-based stopping criterion;
- variance-based global sensitivity analysis using Sobol' total-effect indices.

The current public implementation contains the nonlinear two-degree-of-freedom Bouc–Wen benchmark used to demonstrate the proposed workflow.

## Method summary

Let the source tasks be indexed by \(m=1,\ldots,M\), and let \(t\) denote the target task. Each source-task Gaussian process is first trained independently. Their posterior information is then transferred to the target task through learned nonnegative task weights. The target surrogate is subsequently refined using an adaptive sampling strategy:

1. Generate an initial design and a Monte Carlo candidate set.
2. Construct Voronoi cells around the current training samples.
3. Evaluate the leave-one-out prediction error at each generator.
4. Select the cell associated with the largest error.
5. Add the candidate point farthest from that cell's generator.
6. Update the surrogate and repeat until the normalized error criterion is satisfied.
7. Use the converged surrogate for uncertainty propagation and Sobol' sensitivity analysis.

## Repository contents

```text
ASMLGP-GSA/
├── CVV_GP.m
├── CVV_GPML.m
├── DDOF_Boucwen_2dm.m
├── DDOF_Boucwen_2dm_figure.m
├── N_AML_ak_BW_2d_2dofsys_m1_k1_005.m
├── date_process_sensana_2dof.m
├── date_process_sensanaml_2dof.m
├── model.py
├── optimizer.py
├── utils.py
├── predict_loo_n_1.py
├── predict_loo_n_gp.py
├── predict_ver_n_1.py
├── 2dof_ver_k1_1000.mat
└── Vv2GM.mat
```

### Main MATLAB files

- `N_AML_ak_BW_2d_2dofsys_m1_k1_005.m`  
  Main driver for the two-degree-of-freedom benchmark. It generates source-task and target-task samples, performs adaptive enrichment, calls the Python GP routines, evaluates convergence, and saves the results.

- `DDOF_Boucwen_2dm.m`  
  Nonlinear dynamic solver for the two-degree-of-freedom Bouc–Wen system.

- `DDOF_Boucwen_2dm_figure.m`  
  Variant of the nonlinear solver that additionally returns response histories for visualization.

- `CVV_GP.m`  
  Cross-validation–Voronoi adaptive sampling for the standard single-task GP.

- `CVV_GPML.m`  
  Cross-validation–Voronoi error evaluation using the meta-learned GP.

- `date_process_sensana_2dof.m`  
  Post-processing script for sensitivity analysis with the standard GP surrogate.

- `date_process_sensanaml_2dof.m`  
  Post-processing script for sensitivity analysis with the meta-learned GP surrogate.

### Main Python files

- `model.py`  
  Definition of the scalable meta-learned Gaussian process model and its source-to-target transfer structure.

- `optimizer.py`  
  Marginal-likelihood optimization utilities.

- `utils.py`  
  Supporting GP and tensor utilities.

- `predict_loo_n_1.py`  
  Leave-one-out prediction for the meta-learned GP.

- `predict_loo_n_gp.py`  
  Leave-one-out prediction for the standard GP.

- `predict_ver_n_1.py`  
  Target-task prediction on the validation or sensitivity-analysis sample set.

### Data files

- `Vv2GM.mat`  
  Ground-motion records used by the benchmark.

- `2dof_ver_k1_1000.mat`  
  Validation samples for the two-degree-of-freedom problem.

## Requirements

### MATLAB

- MATLAB R2021b or later is recommended.
- UQLab is required for input-distribution definition and sample generation.
- The scripts were developed on Windows and currently use Windows-style paths.

### Python

Python 3.9 is recommended. The core dependencies are:

```text
torch
botorch
gpytorch
linear-operator
numpy
scipy
pandas
```

A minimal Conda environment can be created with:

```bash
conda create -n asmlgp python=3.9 -y
conda activate asmlgp

pip install torch botorch gpytorch linear-operator numpy scipy pandas
```

Because compatible versions of BoTorch, GPyTorch, PyTorch, and `linear-operator` are important, record the exact versions used in a successful run:

```bash
pip freeze > requirements-lock.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yu-guangquan/ASMLGP-GSA.git
cd ASMLGP-GSA
```

Create the data and result directories expected by the scripts:

```text
meta_tasks/
└── rths_m1/

main_tasks/
└── rths_m1/

result1/
└── AML_m1_k1_005/

sensiti/
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force `
  meta_tasks\rths_m1, `
  main_tasks\rths_m1, `
  result1\AML_m1_k1_005, `
  sensiti
```

## Configuration before running

The MATLAB scripts currently call Python through a hard-coded executable path, for example:

```matlab
system('D:\ProgramData\anaconda3\envs\py391\python.exe predict_ver_n_1.py');
```

Replace this path with the Python executable of your environment:

```matlab
pythonExe = 'C:\path\to\miniconda3\envs\asmlgp\python.exe';
system(sprintf('"%s" predict_ver_n_1.py', pythonExe));
```

The same change should be made wherever the following scripts are called:

```text
predict_loo_n_1.py
predict_loo_n_gp.py
predict_ver_n_1.py
```

Also ensure that MATLAB's current working directory is the repository root, because the code uses relative paths such as:

```text
.\meta_tasks\rths_m1\
.\main_tasks\rths_m1\
.\result1\AML_m1_k1_005\
```

## Running the two-degree-of-freedom benchmark

### 1. Start MATLAB in the repository root

```matlab
cd('path\to\ASMLGP-GSA')
```

### 2. Initialize UQLab

Make sure UQLab is installed and available on the MATLAB path:

```matlab
uqlab
```

### 3. Run the main adaptive-sampling experiment

```matlab
N_AML_ak_BW_2d_2dofsys_m1_k1_005
```

The default script:

- uses one selected stochastic ground-motion record;
- generates \(10^5\) Monte Carlo candidate points;
- starts from five LHS samples;
- constructs source tasks using different Bouc–Wen stiffness parameters;
- adaptively enriches both source and target datasets;
- stops when the normalized leave-one-out error remains below 0.05 for two consecutive iterations;
- repeats the experiment over multiple random seeds;
- saves the resulting datasets, prediction errors, transfer weights, and elapsed time.

### 4. Run sensitivity-analysis post-processing

For the standard GP:

```matlab
date_process_sensana_2dof
```

For the adaptive meta-learned GP:

```matlab
date_process_sensanaml_2dof
```

These scripts use surrogate predictions to estimate total-effect Sobol' indices.

## Key default settings

The main benchmark currently uses:

| Setting | Default |
|---|---:|
| Input dimension | 2 |
| Initial samples | 5 |
| Candidate pool | \(10^5\) |
| Maximum adaptive iterations | 50 |
| Convergence threshold | 0.05 |
| Consecutive satisfied iterations | 2 |
| Repeated random seeds | 10 |
| Target Bouc–Wen stiffness parameter | \(5\times10^5\) |
| Source-task stiffness parameters | \(4\times10^5\), \(3\times10^5\), \(2\times10^5\) |

These values can be changed in `N_AML_ak_BW_2d_2dofsys_m1_k1_005.m`.

## Output files

During execution, intermediate `.mat` files are exchanged between MATLAB and Python.

Typical files include:

```text
meta_tasks/rths_m1/task_1_date.mat
main_tasks/rths_m1/task_main_date.mat
main_tasks/rths_m1/CV_task_main_date.mat
main_tasks/rths_m1/CV_task_test_date.mat
main_tasks/rths_m1/CV_test_prediction.mat
main_tasks/rths_m1/rmse_task_main_date.mat
main_tasks/rths_m1/rmse_task_test_date.mat
main_tasks/rths_m1/test_ver_prediction.mat
```

Final experiment files are written under:

```text
result1/AML_m1_k1_005/
```

Sensitivity-analysis results are written under:

```text
sensiti/
```

## MATLAB–Python data interface

The implementation exchanges data through MATLAB `.mat` files.

For example, the target prediction workflow is:

1. MATLAB saves the current target training set:
   - `Xs_rese`
   - `Ys_rese`
2. MATLAB saves the prediction inputs:
   - `X_ver`
3. Python loads the source-task and target-task data.
4. Python normalizes the input variables using statistics from the target training set.
5. Python trains the meta-learned GP and estimates the transfer weights.
6. Python saves:
   - predictive mean;
   - learned source-task weights.
7. MATLAB loads the predictions and evaluates NRMSE and \(R^2\).

## Reproducibility notes

- The MATLAB driver uses `rng(hh,'twister')` for repeated runs.
- Candidate samples and initial designs are generated with UQLab.
- The current implementation suppresses most Python warnings.
- Numerical results can vary across software versions and hardware.
- Exact package versions, MATLAB version, UQLab version, and random seeds should be reported when reproducing published results.

## Known limitations

- Python executable paths are currently hard-coded in the MATLAB files.
- Several directory paths are fixed in the scripts and must exist before execution.
- The current repository contains the two-degree-of-freedom benchmark rather than a unified command-line interface.
- The scripts are research code and have not yet been packaged as a reusable Python or MATLAB library.
- Input normalization is performed from the current target training set; zero standard deviation in any input dimension must be avoided.
- The current source-data loader in `predict_ver_n_1.py` is configured for one source task and must be adjusted when using multiple simultaneously loaded source tasks.
- Large Monte Carlo candidate sets and repeated leave-one-out fitting can require substantial runtime.

## Citation

When using this repository, please cite the associated paper:

```bibtex
@article{yuASMLGPGSA2026,
  title   = {Adaptive Sampling Meta-Learned Gaussian Processes for Global
             Sensitivity Analysis of Seismic Responses with Partial Structural
             Updating},
  author  = {Yu, Guangquan and Li, Ning and Chen, Cheng and Zhang, Xiaohang
             and Gao, Xiaoshu},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Please update the volume, article number, and DOI after final publication}
}
```

The bibliographic fields above should be replaced with the final Version of Record information once available.

## Contributing

Issues and pull requests that improve reproducibility, portability, documentation, testing, or numerical robustness are welcome. Useful contributions include:

- replacing hard-coded paths with configuration files;
- adding an environment specification;
- adding automated tests;
- providing Linux and macOS support;
- exposing the workflow through a unified driver;
- adding the remaining engineering case studies.

## License

No repository-level license file is currently included. Consequently, default copyright restrictions apply unless the authors specify otherwise. Some individual Python source files contain their own copyright and SPDX notices; those notices remain applicable to those files.

For reuse beyond examination and academic reproducibility, contact the repository authors or wait for an explicit repository-level license.

## Contact

For questions regarding the methodology or implementation, open a GitHub issue in this repository.

Repository: `https://github.com/yu-guangquan/ASMLGP-GSA`
