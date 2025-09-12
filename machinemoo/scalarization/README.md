## 🔧 Base Class: `Scalarization` (moo_scalarization)

This is the base class that you should inherit to implement custom scalarization strategies. It defines the interface for:

- Initializing the scalarization model
- Training the model with a given weight vector
- Returning the trained model and the corresponding objective values (optionally also the gradients from the model)

---

### ✅ How to use

Create your own scalarization class by subclassing `Scalarization`:

```python
from machinemoo import Scalarization

class NewScalarization(Scalarization):
    def __init__(self, num_objs: int, args):
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

If you choose **lower bound estimation with Lipschitz** (`lower_bound_estimate="lipschitz"`),  
you must also implement the method `_compute_lipschitz(self)` in your subclass:

```python
class NewScalarization(Scalarization):
    def _compute_lipschitz(self):
        # Implement your custom lower estimative of the objective values with Lipschitz here
        return self.__objs_lower
```

Alternatively, you can reuse the provided **mixins** for common setups, such as logistic regression with scikit-learn or PyTorch models.

For example, using the PyTorch mixin:

```python
from machinemoo import Scalarization, LipschitzTorchMixin

class NewScalarization(LipschitzTorchMixin, Scalarization):
    def __init__(self, num_objs: int, model):
        super().__init__(num_objs)
        self.model = model
    
    def training(self, weight):
        # Train the PyTorch model here
        return self.model, objs, gradient
```

Example using Logistic Regression from Sci-kit Learn:

```python
from machinemoo import Scalarization, LipschitzRegLoghMixin

class NewScalarization(LipschitzRegLoghMixin, Scalarization):
    def __init__(self, num_objs: int, model):
        super().__init__(num_objs)
        self.model = model
    
    def training(self, weight):
        # Train the PyTorch model here
        return self.model, objs, gradient
```

This way, the Lipschitz lower bound will be computed automatically by the mixin.

### 📘 Example usage

Once you have implemented your scalarization class, you can use it in a multi-objective optimization pipeline like this:

```python
from machinemoo import moo, get_objectives
from my_scalarizations import NewScalarization

w_scalar = NewScalarization(num_objs=4, arg=...)
moopt = moo(w_scalar).mo_optimization(moo_method)
objs = get_objectives(moopt)
```