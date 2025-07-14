# Solving the can problem using Pyomo
###
import numpy as np
import pyomo.environ as pe

###
# parameters
materialAvailable = 10

###
# model
model = pe.ConcreteModel()
model.h = pe.Var(bounds=(0, None))
model.r = pe.Var(bounds=(0, None))
model.con1 = pe.Constraint(
    expr=2 * np.pi * model.r * (model.r + model.h) <= materialAvailable
)

objFun = np.pi * model.r * model.r * model.h
model.obj = pe.Objective(expr=objFun, sense=pe.maximize)

###
ff = open("model_q1.txt", "w")
model.pprint(ostream=ff)
ff.close()
###
solver = pe.SolverFactory("ipopt")
res = solver.solve(model)


## print solution
print(f"Height (h) = {model.h.value:.6f}")
print(f"Radius (r) = {model.r.value:.6f}")
print(f"Max Volume = {pe.value(model.obj):.6f}")
