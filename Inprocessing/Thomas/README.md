# SeldonianML

Python code implementing algorithms that provide high-probability safety guarantees for solving classification, bandit, and off-policy reinforcement learning problems.
For further details, see REFERENCE. 

# Installation

First, install Python 3.x, Numpy (1.16+), and Cython (0.29+).

The remaining dependencies can be installed by executing the following command from the Python directory of : 

	pip install -r requirements.txt

# Usage

Experiments can be launched using the following command from the SeldonianML/Python directory:

    python -m experiments.problem.dataset save_directory <optional arguments>

where problem is either `classification`, `bandit`, or `rl`, and dataset is either `brazil`, `recidivism`, `tutoring`, `credit_assignment`, or `carsim`, depending on the problem type.

Currently, the following experiments are included:

* bandit
	* credit_assignment
	* recidivism
	* tutoring
* classification
	* brazil
	* recidivism
* rl
	* carsim

Alternatively, the experiments from REFERENCE can be executed by running the provided batch file from the Python directory, as follows:

     ./experiments/scripts/science_experiments_brazil.bat
     
Once the experiments complete, the figures found in REFERENCE can be generated using the command, 

     python -m experiments.scripts.science_figures_brazil
     
Once completed, the new figures will be saved to `Python/figures/science/*` by default.

# License

SeldonianML is released under the MIT license.
