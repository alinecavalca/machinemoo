# Machine Learning Multi-Objective Optimization

## Installation

Install required libraries:

```bash
poetry install
```
or

```bash
pip install .
```

## Directories

* `moo/`

Multi-objective methods which return the set of Pareto-optimal solutions.

* `scalarization/`

Scalarization base class.

* `analysis/`

Analysis tools/ a posteriori options.

* `tests/`

Software testing. (TODO)

## Tool methods and returns



## Multi-objective optimization methods

* NISE (Non-Inferior Set Estimation)
* Random Weight

## Usage Tips

* Rennen method is costly for many objectives.
* Random Weights method is slow for many objectives but finds the optimal solution.
* MONISE method is better in some cases but finds sub-optimal solutions.