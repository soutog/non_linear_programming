import numpy as np
import pyomo.environ as pe

# DADOS DE ENTRADA
# pontos A e B (mesmo lado da reta)
x_A, y_A = 1.0, 2.0
x_B, y_B = 5.0, 3.0

# reta refletora:  a x + b y = c
a, b, c = 1.0, 1.0, 4.0

# Modelo Pyomo
mdl = pe.ConcreteModel()


# variáveis: coordenadas de P
mdl.x = pe.Var(initialize=(x_A + x_B) / 2)  # palpite inicial
mdl.y = pe.Var(initialize=(y_A + y_B) / 2)

# restrição: P pertence à reta
mdl.line = pe.Constraint(expr=a * mdl.x + b * mdl.y == c)

# objetivo: minimizar a distância de P até a reta
mdl.obj = pe.Objective(
    expr=pe.sqrt((x_A - mdl.x) ** 2 + (y_A - mdl.y) ** 2)
    + pe.sqrt((x_B - mdl.x) ** 2 + (y_B - mdl.y) ** 2),
    sense=pe.minimize,
)
# ###
ff = open("model_q21.txt", "w")
mdl.pprint(ostream=ff)
ff.close()
# ###
solver = pe.SolverFactory("ipopt")
res = solver.solve(mdl)

# printar resultados
print(f"Coordenadas de P: ({mdl.x.value:.6f}, {mdl.y.value:.6f})")
print(f"Distância mínima = {pe.value(mdl.obj):.6f} unidades")
