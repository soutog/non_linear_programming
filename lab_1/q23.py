import numpy as np
import pyomo.environ as pe

# DADOS
rho = [0.3, 0.5, 0.7]  # densidades pAB, pAC, pBC
ell = [5.0, 7.0, 10.0]  # comprimentos lAB, lAC, lBC  (válidos!)

mAB, mAC, mBC = [r * L for r, L in zip(rho, ell)]

# Modelo Pyomo
mdl = pe.ConcreteModel()

# Variáveis
mdl.xb = pe.Var(domain=pe.Reals, initialize=ell[0])
mdl.yb = pe.Var(domain=pe.Reals, initialize=-1.0)
mdl.xc = pe.Var(domain=pe.Reals, initialize=-ell[1] / 2)
mdl.yc = pe.Var(domain=pe.Reals, initialize=-6.0)

# comprimentos das barras (restrições quadráticas)
mdl.len_AB = pe.Constraint(expr=mdl.xb**2 + mdl.yb**2 == ell[0] ** 2)
mdl.len_AC = pe.Constraint(expr=mdl.xc**2 + mdl.yc**2 == ell[1] ** 2)
mdl.len_BC = pe.Constraint(
    expr=(mdl.xb - mdl.xc) ** 2 + (mdl.yb - mdl.yc) ** 2 == ell[2] ** 2
)

# CM sob o ponto A  >  coordenada-x do CM = 0
mdl.cm_x = pe.Constraint(expr=(mAB + mBC) * mdl.xb + (mAC + mBC) * mdl.xc == 0)

# Objetivo: minimize total pontential energy -> U = m g h = p * l * Ycm
mdl.obj = pe.Objective(
    expr=mAB * mdl.yb + mAC * mdl.yc + mBC * (mdl.yb + mdl.yc), sense=pe.minimize
)


# ---------- SOLVER ----------

ff = open("model_q23.txt", "w")
mdl.pprint(ostream=ff)
ff.close()

solver = pe.SolverFactory("ipopt")
solver.options["tol"] = 1e-9
res = solver.solve(mdl)

# ---------- RESULTADOS ----------
print(f"x_B = {mdl.xb.value:.4f},  y_B = {mdl.yb.value:.4f}")
print(f"x_C = {mdl.xc.value:.4f},  y_C = {mdl.yc.value:.4f}")
print("status:", res.solver.status, " –", res.solver.termination_condition)
print("Energia Minima:", pe.value(mdl.obj))
