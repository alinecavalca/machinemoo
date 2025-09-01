# MachineMOO 🐮

`MachineMOO` is a Python library designed for multi-objective optimization (MOO) with machine learning models. It provides modular components to facilitate scalarization strategies, optimization methods, ensembles, metrics, visualization, and utility functions — all aimed at simplifying research and application of MOO problems.


## 🚀 Key Features

- 🧩 Pluggable scalarization interface
- 📈 Built-in Pareto frontier analysis and visualizations
- 🧪 Support for MOO methods based on weighted sum
- 🔁 Modular ensemble strategies for decision-making

## 📚 Modules

- [`moo`](./moo.md): Multi-objective optimization methods (NISE, MONISE, MOLA, etc.)
- [`moo_scalarization`](./scalarization.md): Create and plug in your own scalarization strategy
- [`analysis`](./analysis.md): Pareto visualization, hypervolume metrics, and model ensembles
- [`utils`](./utils.md): Utility tools like logging