# Solving the facility location problem using Pyomo
###
import numpy as np
import pyomo.environ as pe

# DADOS DE ENTRADA
# coordenadas dos clientes (pontos destino) e demandas
customers = {
    1: {"x": 0.0, "y": 0.0, "d": 20},
    2: {"x": 10.0, "y": 0.0, "d": 25},
    3: {"x": 0.0, "y": 12.0, "d": 15},
    4: {"x": 10.0, "y": 12.0, "d": 10},
}


n = len(customers)
unit_cost = 1.0  # custo unitário de transporte
W = 2  # número de armazéns a instalar

# # Modelo Pyomo
m = pe.ConcreteModel()

# # variáveis de decisão
# # localização dos armazéns (contínuas, não limitadas)
m.I = pe.RangeSet(W)  # armazéns: 1..W
m.J = pe.RangeSet(n)  # clientes : 1..n

# print([customers[j]["d"] for j in m.J])

m.X = pe.Var(m.I)  # coordenada x de cada armazém
m.Y = pe.Var(m.I)  # coordenada y de cada armazém

# # quantidade enviada do armazém i ao cliente j (≥0)
m.q = pe.Var(m.I, m.J, bounds=(0, None))


# # Restrições
def demand_rule(mod, j):
    # para cada cliente j, soma de fluxos que entram ≥ demanda de j
    return sum(mod.q[i, j] for i in mod.I) >= customers[j]["d"]


m.demand = pe.Constraint(m.J, rule=demand_rule)

eps = 1e-6


def cost_expression(mod):
    return unit_cost * sum(
        m.q[i, j]
        * pe.sqrt(
            (m.X[i] - customers[j]["x"]) ** 2 + (m.Y[i] - customers[j]["y"]) ** 2 + eps
        )
        for i in mod.I
        for j in mod.J
    )


m.obj = pe.Objective(rule=cost_expression, sense=pe.minimize)

# solver = pe.SolverFactory("ipopt")
# solver.options["tol"] = 1e-8  # tolerância numérica mais apertada (opcional)
# result = solver.solve(m, tee=False)  # tee=True para log detalhado

# ###
ff = open("model_q12.txt", "w")
m.pprint(ostream=ff)
ff.close()
# ###
solver = pe.SolverFactory("ipopt")
res = solver.solve(m)


# print solution
print("Optimal locations of warehouses:")
for i in m.I:
    print(f"Warehouse {i}: (X, Y) = ({m.X[i].value:.6f}, {m.Y[i].value:.6f})")
print("Optimal shipment quantities:")
for i in m.I:
    for j in m.J:
        if m.q[i, j].value > 0:  # print only positive shipments
            print(f"From Warehouse {i} to Customer {j}: {m.q[i, j].value:.6f}")
print(f"Total transportation cost = {pe.value(m.obj):.6f}")


# print(f"Height (h) = {model.h.value:.6f}")
# print(f"Radius (r) = {model.r.value:.6f}")
# print(f"Max Volume = {pe.value(model.obj):.6f}")
