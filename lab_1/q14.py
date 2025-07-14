import pyomo.environ as pe, math

a, b, c = 1.0, 4.0, 5.0
axes = {1: a, 2: b, 3: c}

I = (1, 2, 3)

m = pe.ConcreteModel()

m.x = pe.Var(I, bounds=(1e-6, None), initialize=lambda m, i: axes[i] / math.sqrt(3))

m.ellipsoid = pe.Constraint(expr=sum(m.x[i] ** 2 / axes[i] ** 2 for i in I) <= 1)

m.obj = pe.Objective(expr=8 * m.x[1] * m.x[2] * m.x[3], sense=pe.maximize)

solver = pe.SolverFactory("ipopt")
solver.options["tol"] = 1e-10
solver.solve(m, tee=False)

print("---- solução ótima ----")
for i, name in zip(I, ("x", "y", "z")):
    print(f"{name}* = {m.x[i].value:.6f}")
print(f"Volume* = {pe.value(m.obj):.6f}")
