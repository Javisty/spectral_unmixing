# Sparse, Smooth and Differentiable Hyperspectral Unmixing using Dispersion Model
Repository for my Research Project 2021-2022 as part of the Master Data Science, about spectral unmixing.

The goal of the project was to re-work the approach to hyperspectral unmixing done in [Differentiable Programming for Hyperspectral Unmixing Using a Physics-based Dispersion Model](https://link.springer.com/chapter/10.1007/978-3-030-58583-9_39) by Janiczek & al., in particular by adding hypotheses on the spatial correlation of the parameters and the smoothness and sparsity of abundances.

Please refer to _RP_unmixing_aymericCOME.pdf_ for further details.

## The model

To summarize, we come up with a differentiable objective function, to minimize using PyTorch autograd fonctionality.

The final term is composed of 3 terms:
- Reconstruction error: the loss associated to the difference between the hyperspectral image and the mixed image, obtained from the learned signatures and abundances
- Regularization on the dispersion model parameters: ensure spatial correlation between hyperpixels (the parameters don't vary too much between adjacent pixels)
- Regularization on the abundances: one term for spatial correlation (as above) and one term for sparsity (few endmembers in each pixel)

## Usage

The _feely.ipynb_ and _synthetic.ipynb_ notebooks contain the experiments detailed in the report, and demonstrate how to generate a model.

The steps are detailed here (N x M image, P endmembers assumed, W wavelengths considered):
- Convert input into torch.tensors with appropriate format: abundance map (N, M, P), mixed image (N, M, W)
- Initialise abundances (torch.tensor)
- Initialise endmembers from InfraRender, into a `DispersionModelDict`
- Initialise the dispersion model, that gives a (N, M, P, W) tensor containing the signatures of each endmember in every pixel (see `EMWrapper` in the notebooks for instance)
- Create the `AbundanceRegularization`, `DispersionRegularization` objects
- Use all these to create the objective function, `SmoothedLinearUnmixing`
- Instantiate the solver with `AutogradDescent` on the objective function
- `solver.fit(...)`

## Requirements

See _pyproject.toml_ file for dependencies.
The project uses the [InfraRender](https://github.com/johnjaniczek/InfraRender) repository as a submodule.

## Setup

Every `import` statement in the ___init.py___ files of the InfraRender folder should be relative (e.g. `from analysis_by_synthesis import ...` -> `from .analysis_by_synthesis import ...`).

`pytest` is the utility chosen to test the integrity of the package. You can run it with `python -m pytest tests/` from the root directory of the project.

## Code structure

Source code is located in the _src_ folder.
_regularization.py_ defines the `torch` objects for the regularization terms of the problem.
_objective.py_ defines the objective function to minimize as a class.
_solver.py_ implements the approach to minimize an objective function.
_utils.py_ contains miscellaneous utilities.

## Notes

- To avoid the differentiability problem at 0 for `sqrt`, I have added a `+ 0.00001`
- I had to do some work-around to make the log-barrier extension backpropagate the gradients correctly. Please look at the comments in the code for further documentation.
- In the notebooks I implement a `EMWrapper` class, as an interface to the collection of EndMembers provided by the InfraRender library, so that calling `forward` on it returns tensors in the expected format.


Do not hesitate to create an issue, I will try to be reactive (statement made in April 2022).
