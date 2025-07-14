import pyomo.environ as pyo


# Create a concrete model
model = pyo.ConcreteModel()
# Define variables
model.x = pyo.Var(within=pyo.NonNegativeReals)
model.y = pyo.Var(within=pyo.NonNegativeReals)

# Define the objective function
model.obj = pyo.Objective(expr=model.x + model.y, sense=pyo.minimize)

# Define constraints
model.con1 = pyo.Constraint(expr=model.x + 2 * model.y >= 4)
model.con2 = pyo.Constraint(expr=3 * model.x + model.y <= 1)

# Select the solver and solve the model
solver = pyo.SolverFactory("ipopt")  # poderia usar "glpk" ou outro solver
result = solver.solve(model)

# Display the results
print(f"Optimal value of x: {model.x.value:.4f}")
print(f"Optimal value of y: {model.y.value:.4f}")
print(f"Objective value: {model.obj():.4f}")
print(f"Solver status: {result.solver.termination_condition}")
print(f"Solver termination condition: {result.solver.termination_condition}")
