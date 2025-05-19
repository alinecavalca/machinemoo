# Machine Learning Multi-Objective Optimization Methods

### Supported methods:
    - NISE
    - Random Weight

#### Nise (Non-Inferior Set Estimation): 
* `About:`

 This approach applies linear approximation to generate an internal approximation (that excludes any solution that might be worse than already found solutions) and an external approximation (that excludes any solution that might be better than optimal solutions). These approximations limit the search space of prototypes that represent the best and worst possible solution for a neighborhood.

 As an iterative technique, it finds a fast approximation by tracing a line between adjacent solutions and using the already computed efficient solutions at each iteration, resulting in the determination of new weights which consist of the one with largest difference between the internal and external approximation.

 To calculate a new efficient solution using the weighted method, this approach uses what is called "neighborhood", which is two efficient solutions. This method proceeds through the following steps.

1. **Initialization:** Generate the first two extreme solutions by finding a Pareto-optimal solution that is also minimum at each objective.

2. **Neighborhood choice:** Determine the next neighborhood to be explored at each iteration.

3. **Calculation of the scalarization weight vector:** Obtain the parameters for the weighted method.

4. **Updating new neighborhoods:** Find new solution and new neighborhoods.

5. **Stopping criterion:** To guarantee the quality of the approximation, a stopping criterion must be established.

 * `Input:`
 * `Output:`

Random Weights:
 * `About:`
 * `Input:`
 * `Output:`
