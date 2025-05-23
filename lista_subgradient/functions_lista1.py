import numpy as np


def f1(xx):
    x = xx[0]
    if x > 0 or x == 0:
        fv = x
        subgrad = 1
    else:
        fv = -x
        subgrad = -1
    return fv, np.array([subgrad])


def f4(x):
    result = abs(x[0] - 1) + abs(x[1] - 2)

    subgrad = [0, 0]  # initialize subgradient
    # Compute subgradient

    if x[0] > 1:
        subgrad[0] = 1
    else:
        subgrad[0] = -1

    if x[1] > 2:
        subgrad[1] = 1
    else:
        subgrad[1] = -1

    return result, np.array(subgrad)
