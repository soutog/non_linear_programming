import math, random
import pyomo.environ as pe

# DADOS
N = 10  # número de cargas
R = 5.0  # raio do anel (m)
k = 1.0  # 1/(4πϵ0) — pode ser 1 porque só muda a escala de U
q = [1.0] * N  # lista de carga

# Modelo Pyomo
mdl = pe.ConcreteModel()

# Variáveis
mdl.I = range(N)
mdl.x = pe.Var(mdl.I, domain=pe.Reals)
mdl.y = pe.Var(mdl.I, domain=pe.Reals)


# restrição do circulo
def circle_rule(m, i):
    return m.x[i] ** 2 + m.y[i] ** 2 == R**2


mdl.circle = pe.Constraint(mdl.I, rule=circle_rule)

# quebra de simetria: fixa a carga 0 em (R,0)
mdl.x[0].fix(R)
mdl.y[0].fix(0)


# F.O.  U = k Σ_{i<j} q_i q_j / r_{ij}
def pair_energy(i, j):
    dx = mdl.x[i] - mdl.x[j]
    dy = mdl.y[i] - mdl.y[j]
    return q[i] * q[j] / pe.sqrt(dx * dx + dy * dy)


mdl.obj = pe.Objective(
    expr=k * sum(pair_energy(i, j) for i in mdl.I for j in mdl.I if i < j),
    sense=pe.minimize,
)

# Palpite inicial: coloca as cargas quase uniformemente distribuídas
for i in mdl.I:
    if i == 0:  # já fixado
        continue
    theta = 2 * math.pi * i / N + random.uniform(-0.05, 0.05)  # quase regular
    mdl.x[i].value = R * math.cos(theta)
    mdl.y[i].value = R * math.sin(theta)


# Solver

ff = open("model_q24.txt", "w")
mdl.pprint(ostream=ff)
ff.close()

solver = pe.SolverFactory("ipopt")
solver.options["tol"] = 1e-9
res = solver.solve(mdl)

# Resultados
print("Status:", res.solver.status, "-", res.solver.termination_condition)
coords = [(pe.value(mdl.x[i]), pe.value(mdl.y[i])) for i in mdl.I]
print("Coordenadas (x,y) das cargas:")
for i, (xi, yi) in enumerate(coords):
    print(f"q{i}: ({xi:.6f}, {yi:.6f})")
print("\nEnergia potencial mínima:", pe.value(mdl.obj))
