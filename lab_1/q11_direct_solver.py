import numpy as np
import scipy.optimize as opt
from math import pi

# ------------------- dados -------------------
A = 10.0  # área de material
x0 = np.array([1.0, 1.0])  # h, r
bnds = ((1e-6, None), (1e-6, None))


# ------------------- objetivo ----------------
def f(x):  # f(h,r)  (negativo p/ maximizar)
    h, r = x
    return -pi * r**2 * h


def grad_f(x):  # ∇f = [∂f/∂h , ∂f/∂r]
    h, r = x
    return np.array([-pi * r**2, -2 * pi * r * h])


# ------------------- restrição ----------------
def g(x):  # g(h,r) ≥ 0   (material que sobra)
    h, r = x
    return A - 2 * pi * r * (r + h)


def grad_g(x):  # ∇g
    h, r = x
    return np.array([-2 * pi * r, -2 * pi * (2 * r + h)])  # ∂g/∂h  # ∂g/∂r


constr = {"type": "ineq", "fun": g, "jac": grad_g}

# ------------------- solver -------------------
res = opt.minimize(
    f,
    x0,
    method="SLSQP",
    jac=grad_f,
    bounds=bnds,
    constraints=[constr],
    options={"ftol": 1e-12, "disp": True},
)

print("\nÓtimo numérico:")
h_opt, r_opt = res.x
print(f"h* = {h_opt:.6f},  r* = {r_opt:.6f}")
print(f"Volume = {-res.fun:.6f}")
