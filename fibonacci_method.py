def recur_fibo(n):
   if n <= 1:
       return 1
   else:
       return(recur_fibo(n-1) + recur_fibo(n-2))

def func(x):
    return pow(x, 4) + 4*(pow(x, 2)) - 32*x + 1

def fibonacci(function):

    accuracy = 0.0001
    a_init = 0
    b_init = 1

    fib = []

    Fn = 0
    n = -1
    while Fn < ((b_init - a_init)/accuracy):
        n += 1
        Fn = recur_fibo(n)
        fib.append(Fn)

    a, b, y, z, fy, fz = [], [], [], [], [], []

    y.append(a_init + fib[n-2] * (b_init - a_init)/fib[n])
    z.append(a_init + fib[n-1] * (b_init - a_init) / fib[n])

    fy.append(function(y[-1]))
    fz.append(function(z[-1]))

    if fy[-1] <= fz[-1]:
        a.append(a_init)
        b.append(z[-1])
    else:
        a.append(y[-1])
        b.append(b_init)

    for k in range(2, n):

        if fy[-1] <= fz[-1]:
            z.append(y[-1])
            fz.append(fy[-1])
            y.append(a[-1] + fib[n-k-3] * (b[-1] - a[-1]) / fib[n-k-1])
            fy.append(function(y[-1]))
        else:
            y.append(z[-1])
            fy.append(fz[-1])
            z.append(a[-1] + fib[n-k-2] * (b[-1] - a[-1]) / fib[n-k-1])
            fz.append(function(z[-1]))

        if fy[-1] <= fz[-1]:
            a.append(a[-1])
            b.append(z[-1])
        else:
            a.append(y[-1])
            b.append(b[-1])

    if fy[-1] <= fz[-1]:
        x = y[-1]
        fx = fy[-1]
    else:
        x = z[-1]
        fx = fz[-1]

    # print('\n# scipy.optimize. #')
    # from scipy.optimize import minimize_scalar
    # res = minimize_scalar(function, bounds=(a_init, b_init), method="bounded")
    # print(f"Minimum is at x = {res.x} with f(x) = {res.fun}")

    return x

res = fibonacci(func)
