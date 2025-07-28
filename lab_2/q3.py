import pyomo.environ as pyo

# ------------------ modelo ------------------
m = pyo.ConcreteModel()

# Variáveis (>=0 só por ilustração; remova bounds se quiser domínio real)
m.x = pyo.Var(bounds=(0, None))
m.y = pyo.Var(bounds=(0, None))

# Objetivo: (x)² + (y-1)²  →  minimizar
m.obj = pyo.Objective(expr=m.x**2 + (m.y - 1) ** 2)

# Restrição:     x² = y
m.con = pyo.Constraint(expr=m.x**2 == m.y)

# ------------------ solução ------------------
solver = pyo.SolverFactory("ipopt")
# Se quiser: solver.options["tol"] = 1e-8  # ou outros parâmetros do IPOPT
res = solver.solve(m, tee=False)

# ------------------ resultados ------------------
print(f"x* = {m.x.value:.6f}")
print(f"y* = {m.y.value:.6f}")
print(f"f(x*,y*) = {pyo.value(m.obj):.6f}")
print(f"Solver status .............: {res.solver.status}")
print(f"Termination condition .....: {res.solver.termination_condition}")
