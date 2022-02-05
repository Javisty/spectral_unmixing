# spectral_unmixing
Repository for my Research Project 2021-2022 as part of the Master Data Science, about spectral unmixing.

## Requirements

See _pyproject.toml_ file for dependencies.
The project uses the [InfraRender](https://github.com/johnjaniczek/InfraRender) repository as a submodule.

## Setup

Every `import` statement in the ___init.py___ files of the InfraRender folder should be relative (e.g. `from analysis_by_synthesis import ...` -> `from .analysis_by_synthesis import ...`).

`pytest` is the utility chosen to test the integrity of the package. You can run it by starting `pytest` from the root directory of the project.

## Code structure

Source code is located in the _src_ folder.
_regularization.py_ defines the `torch` objects for the regularization terms of the problem.
_objective.py_ defines the objective function to minimize as a class.
_solver.py_ implements the approach to minimize an objective function.
_utils.py_ contains miscellaneous utilities.
