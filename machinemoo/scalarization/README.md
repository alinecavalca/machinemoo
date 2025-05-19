## 🔧 Base Class: `Scalarization` (moo_scalarization)

This is the base class that you should inherit to implement custom scalarization strategies. It defines the interface for:

- Initializing the scalarization model
- Training with a given weight vector
- Returning the learned model and the corresponding objective values (optionally also the gradients from the model)

---

### ✅ How to use

You can create your own scalarization class by subclassing `Scalarization`:

```python
from ml_moo import Scalarization

class NewScalarization(Scalarization):
    def __init__(self, model_name: str, num_objs: int, args):
        super().__init__(num_objs)
        self.model = ...  # initialize your model here
        self.args = args    # store any additional arguments
    
    def training(self, weight: list[float]):
        # Train your model based on the weight vector
        # and return (trained_model, objective_values, optional_gradient)

        # Example (pseudocode):
        # self.model.fit(weighted_data)
        # objs = compute_losses(...)
        # gradient = compute_gradient(...)
        return self.model, objs, gradient
```

### 📘 Example usage

Once you have implemented your scalarization class, you can use it in a multi-objective optimization pipeline like this:

```python
from ml_moo import moo, get_objectives
from my_scalarizations import NewScalarization

w_scalar = NewScalarization(model_name="MLP", num_objs=3, arg=...)
moopt = moo(w_scalar).mo_optimization(moo_method, opt_params)
objs = get_objectives(moopt)
```