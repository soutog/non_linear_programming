import pyomo.environ as pe


nodes = ["s", "a", "b", "c", "d", "t"]
arcs_cap = {
    ("s", "a"): 16,
    ("s", "c"): 13,
    ("a", "c"): 10,
    ("c", "a"): 4,
    ("a", "b"): 12,
    ("c", "b"): 9,
    ("b", "d"): 7,
    ("c", "d"): 14,
    ("b", "t"): 20,
    ("d", "t"): 4,
}
N = nodes
A = list(arcs_cap.keys())
s, t = "s", "t"


m = pe.ConcreteModel()

m.N = pe.Set(initialize=N)
m.A = pe.Set(within=m.N * m.N, initialize=A)

m.u = pe.Param(m.A, initialize=arcs_cap)  # capacidade c_ij
m.f = pe.Var(m.A, bounds=(0, None))  # fluxo f_ij   (contínuo)


def cap_rule(mod, i, j):
    return mod.f[i, j] <= mod.u[i, j]


m.cap = pe.Constraint(m.A, rule=cap_rule)


def balance_rule(mod, v):
    outflow = sum(mod.f[v, j] for j in mod.N if (v, j) in mod.A)
    inflow = sum(mod.f[j, v] for j in mod.N if (j, v) in mod.A)
    if v == s:
        return outflow - inflow == mod.F  # variável auxiliar F
    elif v == t:
        return inflow - outflow == mod.F  # mesma F
    else:  # vértices internos
        return outflow - inflow == 0


m.F = pe.Var()  # valor do fluxo total
m.balance = pe.Constraint(m.N, rule=balance_rule)

m.obj = pe.Objective(expr=m.F, sense=pe.maximize)

solver = pe.SolverFactory("ipopt")
solver.options["tol"] = 1e-8
result = solver.solve(m, tee=False)

print(f"\nFluxo máximo s→t  = {m.F.value:.0f}\n")

for i, j in m.A:
    if m.f[i, j].value > 1e-8:
        print(f"f[{i},{j}] = {m.f[i,j].value:.0f}  (cap {m.u[i,j]})")
