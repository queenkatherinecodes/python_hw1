import numpy as np


def quadratic_example_template(x, Q, is_newton):
    Q_x = np.dot(Q, x)
    f = np.dot(x.T, Q_x)
    g = 2 * Q_x
    if is_newton:
        return f, g, 2 * Q
    return f, g


def quadratic_example_1(x, is_newton):
    Q = np.array([[1,0],[0,1]])
    return quadratic_example_template(x, Q, is_newton)


def quadratic_example_2(x, is_newton):
    Q = np.array([[1,0],[0,100]])
    return quadratic_example_template(x, Q, is_newton)


def quadratic_example_3(x, is_newton):
    A = np.array([[(np.sqrt(3)/2), -.5],[.5, (np.sqrt(3)/2)]])
    Q = np.dot(A.T, np.dot(np.array([[100,0],[0,1]]), A))
    return quadratic_example_template(x, Q, is_newton)


def rosenbrock_function(x, is_newton):
    x_1 = x[0]
    x_2 = x[1]
    f = 100 * ((x_2 - (x_1 ** 2)) ** 2) + (1 - x_1)**2
    g = np.array([400 * (x_1 ** 3) - 400 * x_1 * x_2 + 2 * x_1 - 2, 200 * x_2 - 200 * (x_1 ** 2)])
    if is_newton:
        h = np.array([[1200 * (x_1 ** 2) - 400 * x_2 + 2, -400 * x_1],[-400 * x_1, 200]])
        return f, g, h
    return f, g


def linear_function(x, is_newton):
    a = np.array([6.0,9.0])
    f = np.dot(a, x)
    g = a
    if is_newton:
        h = np.array([[0.0,0.0],[0.0,0.0]])
        return f, g, h
    return f, g


def exponential_function(x, is_newton):
    x_1 = x[0]
    x_2 = x[1]
    f = np.exp(x_1 + 3 * x_2 - .1) + np.exp(x_1 - 3 * x_2 - .1) + np.exp(-x_1 - .1)
    g = np.array([np.exp(x_1 + 3 * x_2 - .1) + np.exp(x_1 - 3 * x_2 - .1) - np.exp(-x_1 - .1), 3 * np.exp(x_1 + 3 * x_2 - .1) - 3 * np.exp(x_1 - 3 * x_2 - .1)])
    if is_newton:
        h = np.array([[f, 3 * np.exp(x_1 + 3 * x_2 - .1) - 3 * np.exp(x_1 - 3 * x_2 - .1)],[3 * np.exp(x_1 + 3 * x_2 - .1) - 3 * np.exp(x_1 - 3 * x_2 - .1), 9 * np.exp(x_1 + 3 * x_2 - .1) + 9 * np.exp(x_1 - 3 * x_2 - .1)]])
        return f, g, h
    return f, g
