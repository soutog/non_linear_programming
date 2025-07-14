import numpy as np
import pyomo.environ as pe

# DADOS DE ENTRADA (barra AB de comprimento l)
mass = 20.0  # kg
g = 9.81  # m/s²
l = 10.0  # comprimento da barra
a = 3.0  # distância de A até o ponto de fixação C

d = abs(l / 2 - a)  # distância CM → C   (d > 0 se C ≠ centro de massa)

# Modelo Pyomo
mdl = pe.ConcreteModel()

# variável: ângulo da barra em relação à horizontal
mdl.theta = pe.Var(bounds=(-np.pi / 2, np.pi / 2), initialize=0.2)  # rad

# energia potencial relativa:  U = m g (-d sin θ)
mdl.obj = pe.Objective(
    expr=-mass * g * d * pe.sin(mdl.theta), sense=pe.minimize  # minimizar
)
# ###
ff = open("model_q22.txt", "w")
mdl.pprint(ostream=ff)
ff.close()
# ###
solver = pe.SolverFactory("ipopt")
res = solver.solve(mdl)

# printar resultados
print(f"Ângulo θ = {mdl.theta.value:.6f} rad")
print(f"Ângulo θ = {np.degrees(mdl.theta.value):.2f} graus")
print(f"Energia potencial mínima = {pe.value(mdl.obj):.6f} J")
