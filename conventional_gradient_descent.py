import numpy as np
from fibonacci_method import fibonacci
from primal_simplex import simplex_method
from copy import deepcopy


def partial_derivatives(f, x0, y0):
    h = 1e-10  # small increment
    f_x = (f(x0 + h, y0) - f(x0, y0))/h
    f_y = (f(x0, y0 + h) - f(x0, y0))/h
    return f_x, f_y


def get_beta(x, grad):

    def q(beta):
        new_x1 = x[0] - beta * grad[0]
        new_x2 = x[1] - beta * grad[1]

        return f(new_x1, new_x2)

    beta = fibonacci(q)
    return beta

# My task
epsilon = 0.00000001
x_start = [0, 0]

def f(x1, x2):
    return 2*x1**2 - x2**2 + 20*x1 - 5*x2 + 8

A = [[1, 8, 1, 0],
    [2, 1, 0, 1]]
P = [5, 10]


def conventional_gradient():
    x_curr = x_start
    x_prev = [val + 20*epsilon for val in x_curr]

    itr = 0
    while abs(f(*x_curr) - f(*x_prev)) > epsilon:
        itr += 1
        x_prev = x_curr
        dx1, dx2 = partial_derivatives(f, x_curr[0], x_curr[1])
        gradient = [dx1, dx2]
        c = gradient + [0] * len(A)

        x = simplex_method(deepcopy(A), deepcopy(P), c)
        y = x[:len(x_start)]

        h = [y[i] - x_curr[i] for i in range(len(y))]

        beta = get_beta(x_curr, gradient)

        x_curr = [x_curr[i] + beta * h[i] for i in range(len(x_curr))]
        print(f"{itr} | f: {f(*x_curr):.20f}, x: {[f'{val:.20f}' for val in x_curr]}")

conventional_gradient()


from scipy.optimize import minimize, LinearConstraint
print("\n\n## check results ##")
def f_check(X):
    return f(*X)
constraint = LinearConstraint(
    [row[:2] for row in A], 0, ub=P)
res = minimize(fun=f_check, x0=x_start, constraints=constraint,
               tol=epsilon, bounds=((0, None), (0, None)))
print(f"Result: {res.fun:.20f}, X: {[f'{val:.20f}' for val in res.x]}")
