# MachineMOO 🐮

`MachineMOO` is a Python library designed for multi-objective optimization (MOO) with machine learning models. It provides modular components to facilitate scalarization strategies, optimization methods, ensembles, metrics, visualization, and utility functions — all aimed at simplifying research and application of MOO problems.


## 🚀 Key Features

- 🧩 Pluggable scalarization interface
- 📈 Built-in Pareto frontier analysis and visualizations
- 🧪 Support for MOO methods based on weighted sum
- 🔁 Modular strategies for decision-making

## 📚 Modules

- [`moo`](./moo.md): Multi-objective optimization methods (MOLA, NISE, MONISE, Random Weights)
- [`moo_scalarization`](./scalarization.md): Create and plug in your own scalarization strategy
- [`analysis`](./analysis.md): Pareto visualization, hypervolume metrics, and strategies for decision-making
- [`utils`](./utils.md): Utility tools like centralized logging configuration to streamline development and debugging.