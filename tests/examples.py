import numpy as np


def quadratic_example_1(x, is_newton):
    Q = np.array([[1,0],[0,1]])
    Q_x = np.dot(Q, x)
    f = np.dot(x.T, Q_x)
    g = 2 * Q_x
    if is_newton:
        return f, g, 2 * Q
    return f, g