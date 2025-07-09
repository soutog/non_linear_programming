import numpy as np
import scipy.optimize as opt

# solving by direct integration with the solver (Can problem)
# define a pattern for the decision variable as vector
# x = [h,r]


def objFun(x):
    val = -np.pi * x[1] ** 2 * x[0]  # negative to maximize
    return val


# coding the constraints

materialAvailable = 10  # available material in kg


def con1(x):
    val = materialAvailable - (2 * np.pi * x[1] ** 2 + 2 * np.pi * x[1] * x[0])
    return val


# calling the solver

x0 = np.array([1, 1])  # initial guess for h and r
res = opt.minimize(
    objFun,
    x0,
    bounds=((0, np.infty), (0, np.infty)),
    constraints={"type": "ineq", "fun": con1},
    method="SLSQP",
)

print(res)
