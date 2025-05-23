import numpy as np

def minimize(f, x0, obj_tol, param_tol, max_iter, type='gradient'):
    x_k = x0
    for i in range(max_iter):
        x = np.array(x_k, dtype=float)
        if type == "gradient":
            f_x, g_x = f(x, False)
            p_k = -g_x
            alpha = set_alpha(f, p_k, x)
        else:
            f_x, g_x, h_x = f(x, True)
            p_k = np.linalg.solve(h_x, -g_x)
            alpha = set_alpha(f, p_k, x)
        print(f'Iteration {i:3d}: x = {x}, f(x) = {f_x:.6f}')
        x_k = x + alpha * p_k
        f_x_k, _ = f(x_k, False)
        if can_terminate(x_k, x, f_x_k, f_x, obj_tol, param_tol):
            print(f'Success! Final iteration: x = {x_k}, f(x) = {f_x_k:.6f}')
            return x_k, f_x_k, True
    print(f'Failure... Final iteration: x = {x_k}, f(x) = {f_x_k:.6f}')
    return x_k, f_x_k, False


def set_alpha(f, p_k, x, rho=.5, c=.01):
    f_x_k, g_x_k = f(x, False)
    alpha = 1
    small, _ = f(x + alpha*p_k, False)
    big = f_x_k + c * np.dot(g_x_k, p_k) * alpha
    while small > big:
        alpha = rho * alpha
        small, _ = f(x + alpha * p_k, False)
        big = f_x_k + c * np.dot(g_x_k, p_k) * alpha
    return alpha


def can_terminate(x_k, x, f_x_k, f_x, obj_tol, param_tol):
    obj_change = abs(f_x_k, f_x)
    param_change = np.linalg.norm(x_k - x)
    return obj_change < obj_tol or param_change < param_tol
