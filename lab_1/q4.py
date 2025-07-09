# Solving the kepler's problem using Pyomo
import pyomo.environ as pe

model = pe.ConcreteModel("Kepler's Problem")
I = list(range(1, 4))  # Indices for the three planets
param = {1: 1, 2: 4, 3: 5}

model.x = pe.Var(I, bounds=(0, None))
model.con1 = pe.Constraint(expr=sum(model.x[i] ** 2 / param[i] ** 2 for i in I))

objFUn = 8 * model.x[1] * model.x[2] * model.x[3]
model.obj = pe.Objective(expr=objFUn, sense=pe.maximize)

ff = open("kepler.txt", "w")
model.pprint(ostream=ff)
ff.close()

solver = pe.SolverFactory("ipopt")
res = solver.solve(model, tee=True)

for i in I:
    print(f"x[{i}] = {model.x[i].value:.6f}")

print(f"Objective value = {model.obj():.6f}")
