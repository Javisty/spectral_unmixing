# spectral_unmixing
Repository for my Research Project 2021-2022 as part of the Master Data Science, about spectral unmixing.

## Requirements

See _pyproject.toml_ file for dependencies.
The project uses the [InfraRender](https://github.com/johnjaniczek/InfraRender) repository as a submodule.

## Setup

Every `import` statement in the ___init.py___ files of the InfraRender folder should be relative (e.g. `from analysis_by_synthesis import ...` -> `from .analysis_by_synthesis import ...`).

`pytest` is the utility chosen to test the integrity of the package. You can run it by starting `pytest` from the root directory of the project.

