# Uncertainty quantification for airflow-exposure model in Vadere

This repository contains the numerical experiments and analysis used in the paper on uncertainty quantification (UQ) for an airflow-exposure model implemented in [Vadere](https://gitlab.lrz.de/vadere/vadere).

The project performs forward uncertainty propagation and global sensitivity analysis for airflow-related model parameters. The workflow relies on the [SALib](https://salib.readthedocs.io/en/latest/) library for sampling and analysis and uses the [SUQ-Controller](https://gitlab.lrz.de/vadere/suq-controller "SUQ-Controller") to run simulations of the model in parallel.

This repository is based on and extends the uncertainty quantification framework originally developed by Simon Rahn:
https://gitlab.lrz.de/vadere/infection-model-uq

The original framework was adapted to support the experiments presented in the paper.

## Main changes compared to the original framework

The UQ package included in this repository ("vimuq" folder) is a lightly modified version of the original implementation. The main extension is:

- Support for batch processing of multiple scenario files via scenario directories, instead of requiring a single scenario file as input.

This allows running UQ studies over sets of scenarios in a single experiment.

Apart from this extension, the core workflow and methods of the original framework remain unchanged.

## Repository structure

The repository is organized as follows (mostly keeping the structure of infection-model-uq):

<pre>
.
├── suq-controller              # git submodule
├── vadere                      # git submodule
├── vimuq                       # Modified UQ framework (based on infection-model-uq)
├── tools/ContinuousIntegration # script for running all unit tests
└── experiments                 # Numerical experiments used in the paper
    ├── analysis                # Defines the analysis of the simulation runs (second step)
    │    ├── <var>EXPERIMENT_XY</var>.py   # Can be run from terminal or IDE 
    │    └── ...  
    ├── definition              # Definition of numerical experiments (first step)
    │    ├── <var>EXPERIMENT_XY</var>.py   # Can be run from terminal or IDE 
    │    └── ...  
    ├── model  
    │   └── vadere              # Vadere model 
    │       ├── scenarios       # Scenario files (`.scenario` files)
    │       ├── simulator       # Vadere console JAR files, e.g., `vadere-console.jar`
    │       └── vadere.project  # Vadere project file (optional)
    ├── output/<var>EXPERIMENT_XY</var>/   # Output for each numerical experiment
    │   ├── scenario            # Contains a copy of the scenario file or folder 
    │   ├── simulation          # (Raw) simulation data
    │   └── summary             # Dumped (`.pkl`) experiment setup, processed simulation data, analysis
    │                           # results, and meta data
    ├── plot                    # Code for visualizing the Sobol' indices and Monte Carlo results (third step)
    └── validation              # includes the restaurant scenario used for validation and its simulation results, 
                                # as well as a script for extracting the number of highly exposed agents
</pre>

## Uncertainty quantification methods

The framework currently supports the following UQ methods.

- Forward uncertainty propagation: Monte Carlo sampling
- Global sensitivity analysis: Sobol' sensitivity indices (implemented via SALib)

## Supported parameter distributions

The following probability distributions can be used for model parameters:
- Uniform (used in our experiments)
- Normal
- Truncated normal
- Log-normal (defined in log-space)


## Setup 

### System requirements

- Python 3.8
- see requirements.txt

### Installation

Clone the repository including submodules:
```
git clone git@gitlab.lrz.de:vadere/infection-model-uq.git
git submodule update --init --recursive
```
Create and activate a virtual environment: 
```
sudo apt install python3.8-venv
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
``` 
Install dependencies: 
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
``` 
Install the UQ package in development mode: 
``` 
pip install -e .
``` 
### Vadere setup 

For installing Vadere, follow the procedure described in https://gitlab.lrz.de/vadere/vadere. 
Specifically, for setting up the airflow model, follow the steps described in VadereSimulator/src/org/vadere/simulator/models/airflow/python/README.md. 

## Running experiments 

The workflow typically consists of two main steps:
1. Run experiments : Evaluate the model for sampled parameter configurations.
2. Analyze results : Compute statistics and sensitivity indices from the simulation outputs.


## Authors and acknowledgment

The uncertainty quantification framework used in this repository is based on the work of 
Simon Rahn ([infection-model-uq project](https://gitlab.lrz.de/vadere/infection-model-uq))

The repository was extended for the experiments conducted in the associated research work by Sophia Johanna Wagner. 


## License 

This project is licensed under the GNU Lesser General Public License (LGPL-3.0).
