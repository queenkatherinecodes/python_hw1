import numpy as np


def minimize(f, x0, obj_tol, param_tol, max_iter, type='gradient'):
    x_k = x0
    x_history = []
    f_history = []
    try:
        for i in range(max_iter):
            x = np.array(x_k, dtype=float)
            if type == "gradient":
                f_x, g_x = f(x, False)
                p_k = -g_x
                alpha = set_alpha(f, p_k, x)
                can_terminate_newton = False
            else:
                f_x, g_x, h_x = f(x, True)
                p_k = np.linalg.solve(h_x, -g_x)
                alpha = set_alpha(f, p_k, x)
                can_terminate_newton = .5 * np.dot(p_k, np.dot(h_x, p_k)) < obj_tol
            x_history.append(x)
            f_history.append(f_x)
            print(f'Iteration {i:3d}: x = {x}, f(x) = {f_x:.6f}')
            x_k = x + alpha * p_k
            f_x_k, _ = f(x_k, False)
            if can_terminate(x_k, x, f_x_k, f_x, obj_tol, param_tol) or can_terminate_newton:
                x_history.append(x_k)
                f_history.append(f_x_k)
                print(f'Success! Final iteration: x = {x_k}, f(x) = {f_x_k:.6f}')
                return x_k, f_x_k, True, x_history, f_history
        x_history.append(x_k)
        f_history.append(f_x_k)
        print(f'Failure... Final iteration: x = {x_k}, f(x) = {f_x_k:.6f}')
        return x_k, f_x_k, False, x_history, f_history
    except np.linalg.LinAlgError as e:
        print(f'Failure - Linear algebra error at iteration {i}: {str(e)}')
        if x_history:
            return x_history[-1], f_history[-1], False, x_history, f_history
        else:
            f_x, _ = f(x0, False)
            return x0, f_x, False, [x0], [f_x]


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
    obj_change = abs(f_x_k - f_x)
    param_change = np.linalg.norm(x_k - x)
    return obj_change < obj_tol or param_change < param_tol
