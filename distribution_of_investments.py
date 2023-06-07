import numpy as np

companies = [
    [(2, 0.5), (2, 0.4), (4, 1.4), (4, 1.2), (5, 1.3)],
    [(3, 0.8), (3, 0.6), (4, 1.4), (2, 6), (3, 0.8)],
    [(1, 3.0), (2, 0.4), (4, 1.4), (3, 0.9), (5, 1.3)]
]

budget_init = 2

max_cost = max(project[0] for company in companies for project in company)

def function(array):
    max_values = []
    for i, row in enumerate(array):
        max_value = max(row)
        col = row.index(max_value)
        max_values.append((max_value, col))
    return max_values

f_ = []

for company in reversed(companies):
    A = [[0 for j in range(max_cost+1)] for i in range(budget_init+1)]
    f_prev = f_[-1] if f_ else [(0, 0)]*len(A)

    for i in range(len(A)):
        for j in range(len(A[0])):

            if j > i:
                continue

            for cost, income in company:
                if cost == j:
                    # print(f_prev[i-j][0])
                    A[i][j] = max(income, A[i][j])
            A[i][j] += f_prev[i - j][0]
    # print(np.array(A))
    f_.append(function(A))
# print(f_)

budget = budget_init
total_profit = 0

for i, state in enumerate(reversed(f_)):
    project_cost = state[budget][1]
    project = (None, 0, 0)
    for j, (cost, profit) in enumerate(companies[i]):
        if cost == project_cost and profit >= project[1]:

            project = (j+1, profit, cost)
    budget -= project_cost
    total_profit += project[1]

    print(f'Project: {project[0]}, company:{i+1}, profit: {project[1]}, cost: {project[2]}')
print(f'Total profit: {total_profit}')









