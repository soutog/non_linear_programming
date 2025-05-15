import numpy as np


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
