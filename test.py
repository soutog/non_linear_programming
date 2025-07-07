from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Objective,
    SolverFactory,
    value,
    minimize,
)

# -------------------------------------------------
# 1) Modelo de exemplo: minimização de (x-1)² + (y-2)²
#    sujeito a x + y = 20  e  x,y >= 0
#    (solução ótima: x=1, y=19, f=289)
# -------------------------------------------------

m = ConcreteModel()

# Variáveis (domínio: reais não-negativos)
m.x = Var(bounds=(0, None), initialize=0.0)
m.y = Var(bounds=(0, None), initialize=0.0)

# Restrição linear
m.eq_total = Constraint(expr=m.x + m.y == 20)

# Função-objetivo
m.obj = Objective(expr=(m.x - 1) ** 2 + (m.y - 2) ** 2, sense=minimize)

# -------------------------------------------------
# 2) Chamada ao Ipopt
# -------------------------------------------------

solver = SolverFactory("ipopt")  # Usa o binário 'ipopt' encontrado no PATH
# NENHUMA opção 'linear_solver' é necessária; MUMPS já é o padrão

results = solver.solve(m, tee=True)  # tee=True imprime o log completo do Ipopt

# -------------------------------------------------
# 3) Exibe o resultado de forma amigável
# -------------------------------------------------

print("\n===== RESUMO =====")
print("Status :", results.solver.termination_message)
print(f"x* = {value(m.x):.6f}")
print(f"y* = {value(m.y):.6f}")
print(f"f(x*,y*) = {value(m.obj):.6f}")
