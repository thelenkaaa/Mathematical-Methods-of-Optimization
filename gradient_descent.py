from fibonacci_method import fibonacci
from scipy.optimize import minimize
from copy import deepcopy


def f(x1, x2):
    return 4 * x1 ** 2 - 4 * x1 * x2 + 5 * x2 ** 2 - x1 - x2

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


eps = 0.000000001
x_init = [0, 0]
x0 = deepcopy(x_init)
x1 = [None, None]

itr = 1
while True:

    dx1, dx2 = partial_derivatives(f, x0[0], x0[1])

    gradient = [dx1, dx2]

    beta = get_beta(x0, gradient)

    x1[0] = x0[0] - beta * gradient[0]
    x1[1] = x0[1] - beta * gradient[1]

    print(itr, x1, f(*x1))
    itr += 1

    if abs(f(*x1) - f(*x0)) <= eps:
        break

    x0[0] = x1[0]
    x0[1] = x1[1]


def f_check(X):
    return f(*X)

print("#check with scipy.minimize#")
print(minimize(f_check, x_init))
