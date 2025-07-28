import pyomo.environ as pyo

# ------------------ modelo ------------------
m = pyo.ConcreteModel()

# Variáveis (>=0 só por ilustração; remova bounds se quiser domínio real)
m.x = pyo.Var(bounds=(0, None))
m.y = pyo.Var(bounds=(0, None))

# Objetivo: (x‑2)² + y²  →  minimizar
m.obj = pyo.Objective(expr=(m.x - 2) ** 2 + m.y**2)

# Restrição:     x = sin(y)
m.con = pyo.Constraint(expr=m.x == pyo.sin(m.y))

# (opcional) ponto inicial ajuda o IPOPT a fugir de mínimos locais ruins
# m.x.set_value(0.5)
# m.y.set_value(0.5)

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
