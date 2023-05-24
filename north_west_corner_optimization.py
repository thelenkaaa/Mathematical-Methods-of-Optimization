import pulp
import numpy as np

def equality():
    if sum(stock) != sum(requirements):
        raise ValueError('Sum of needs is not equal to sum of supplies')


def check_oporn_plan(X):
    count = 0
    m = len(X)
    n = len(X[0])
    for k in range(len(X)):
        for t in range(len(X[0])):
            if X[k][t] != -1:
                count += 1
    if count == (m + n - 1):
        print("this plan is supportive!")
        return True
    print("this plan is NOT supportive!")
    return False

def total_cost():
    cost = 0
    for i in range(len(X)):
        for j in range(len(X[0])):
            if X[i][j] != -1:
                cost += X[i][j] * C[i][j]
    return cost

def north_west_corner_method():
    i, j = 0, 0
    m = len(C) - 1
    n = len(C[0]) - 1
    X[i][j] = min(stock[i], requirements[j])
    while i <= len(C) and j <= len(C[0]):
        if requirements[j] > stock[i]:
            X[i][j] = min(stock[i], requirements[j])
            requirements[j] -= stock[i]
            stock[i] = 0
            i += 1

        elif stock[i] > requirements[j]:
            X[i][j] = min(stock[i], requirements[j])
            stock[i] -= requirements[j]
            requirements[j] = 0
            j += 1

        elif stock[i] == requirements[j] and i == m and j == n:
            X[i][j] = requirements[j]
            stock[i], requirements[j] = 0, 0
            # print(np.array(X))
            check_oporn_plan(X)
            total = total_cost()
            print(f"Total cost: {total}")
            break
        else:
            X[i][j] = requirements[j]
            stock[i], requirements[j] = 0, 0
            if i == m:
                j += 1
                X[i][j] = 0
                i += 1
            else:
                i += 1
                X[i][j] = 0
                j += 1
    print(X)
    return X

def calculate_delta(u, v):
    delta = [[0 for i in range(n)] for j in range(m)]
    for i in range(m):
        for j in range(n):
            if X[i][j] == -1:
                delta[i][j] = u[i] + v[j] - C[i][j]
    return delta

def get_top_vertex(deltas):
    max_delta = max(max(row) for row in deltas)
    max_delta_idx = []
    for i in range(m):
        if max_delta in deltas[i]:
            max_delta_idx.append(i)
            max_delta_idx.append(deltas[i].index(max_delta))
    max_delta_row = max_delta_idx[0]
    max_delta_col = max_delta_idx[1]
    return max_delta_row, max_delta_col

def get_vertexes_for_dfs():
    tree = list(zip(*np.where(np.array(X) != -1)))
    return tree

def check_free_cells(delta):
    count = 0
    for i in range(m):
        for j in range(n):
            if delta[i][j] > 0:
                count += 1
    if count > 0:
        return False
    return True

def get_potentials(cost, x_indixes):
    m = len(cost)
    n = len(cost[0])

    u = [None] * m
    v = [None] * n
    u[0] = 0

    while None in u or None in v:
        for idx in range(len(x_indixes)):
            row = x_indixes[idx][0]
            col = x_indixes[idx][1]
            if not u[row] is None:
                v[col] = cost[row][col] - u[row]
            elif not v[col] is None:
                u[row] = cost[row][col] - v[col]

    return u, v

def update_table(cycle, tree):
    # Update the X table
    X[cycle[0][0]][cycle[0][1]] = 0
    min_val = float("inf")
    min_i = None
    min_j = None
    for i in range(m):
        for j in range(n):
            if (i, j) not in cycle[1::2]:
                continue
            if X[i][j] < min_val:
                min_val = X[i][j]
                min_i = i
                min_j = j
    out_i = min_i
    out_j = min_j
    amount = X[out_i][out_j]
    for index, (i, j) in enumerate(cycle):
        if index % 2 == 0:
            X[i][j] += amount
        else:
            X[i][j] -= amount
    X[out_i][out_j] = -1

    # Update the tree list
    for i in range(len(tree)):
        if tree[i] == (out_i, out_j):
            tree[i] = (cycle[0][0], cycle[0][1])
            break
    return X, tree

# Define a function to find a cycle in a graph
def get_cycle(starting_point, x_indices):
    # Add the starting point to the list of vertices to be searched
    vertices = x_indices + [starting_point]

    # Find the neighbors of the starting point
    neighbors = []
    for i in range(len(vertices)):
        if vertices[i][0] == starting_point[0] or vertices[i][1] == starting_point[1] and vertices[i] != starting_point:
            neighbors.append(vertices[i])

    # Iterate over the neighbors and search for a cycle
    for i in range(len(neighbors)):
        # Initialize a dictionary to keep track of which vertices have been visited
        is_visited = {}
        for j in range(len(vertices)):
            is_visited[vertices[j]] = False

        # Use depth-first search to find a cycle
        is_cycle, cycle = dfs(is_visited, vertices, neighbors[i], starting_point, came_by_along_row=(neighbors[i][0] == starting_point[0]))

        # If a cycle is found, break out of the loop and return the cycle
        if is_cycle:
            break

    return cycle


# Define a helper function to perform depth-first search
def dfs(is_visited, vertices, current_vertex, cycle_start, came_by_along_row):
    is_cycle = False
    cycle = []

    # Check if the current vertex has already been visited
    if is_visited[current_vertex]:
        return is_cycle, cycle

    # Mark the current vertex as visited
    is_visited[current_vertex] = True

    # If the current vertex is the same as the starting and it's not the first time we visited it, cycle has been found
    if current_vertex == cycle_start:
        is_cycle = True
        cycle.append(current_vertex)

    # Otherwise, continue searching
    else:
        neighbors = []
        # Find the neighbors of the current vertex
        if came_by_along_row:
            for i in range(len(vertices)):
                if vertices[i][1] == current_vertex[1]:
                    neighbors.append(vertices[i])
        else:
            for i in range(len(vertices)):
                if vertices[i][0] == current_vertex[0]:
                    neighbors.append(vertices[i])

        # Iterate over the neighbors and search for a cycle
        for i in range(len(neighbors)):
            is_cycle, cycle = dfs(is_visited, vertices, neighbors[i], cycle_start, came_by_along_row = not came_by_along_row)

            # If a cycle is found, add the current vertex to the cycle and break out of the loop
            if is_cycle:
                cycle.append(current_vertex)
                break

    return is_cycle, cycle

def method_of_potencials(indexes):
    while True:
        print()
        tree = get_vertexes_for_dfs()
        u, v = get_potentials(C, tree)
        delta = calculate_delta(u, v)
        if check_free_cells(delta):
            break
        else:
            print('There are deltas that are positive')
            max_delta_row, max_delta_col = get_top_vertex(delta)
            tree = get_vertexes_for_dfs()
            cycle = get_cycle((max_delta_row, max_delta_col), tree)
            X, tree = update_table(cycle, tree)
            print(X)
            total = total_cost()
            print(f'Total cost: {total}')

def show_right_answer(supply, demand, cost):
    A = np.array(stock, dtype=np.float32)
    B = np.array(requirements, dtype=np.float32)
    C = np.array(cost, dtype=np.float32)

    # Create the LP problem
    transport_lp = pulp.LpProblem('Transportation', pulp.LpMinimize)

    # Define the decision variables
    rows = len(supply)
    cols = len(demand)

    variables = pulp.LpVariable.dicts("Transport", ((i, j) for i in range(rows) for j in range(cols)), lowBound=0, cat='Continuous')

    # Define the objective function
    transport_lp += pulp.lpSum([variables[i, j] * cost[i][j] for i in range(rows) for j in range(cols)])

    # Define the supply constraints
    for i in range(rows):
        transport_lp += pulp.lpSum([variables[i, j] for j in range(cols)]) <= supply[i]

    # Define the demand constraints
    for j in range(cols):
        transport_lp += pulp.lpSum([variables[i, j] for i in range(rows)]) == demand[j]

    # Solve the LP problem
    status = transport_lp.solve(pulp.PULP_CBC_CMD(msg=0))

    # Print the results
    print("Status: ", pulp.LpStatus[status])
    print("Minimum Cost: ", pulp.value(transport_lp.objective))
    tableau = np.array([v.varValue for v in transport_lp.variables()]).reshape(C.shape)
    print(f'Tableau: \n{tableau}')

if __name__ == '__main__':
    stock = [60, 40, 100, 50]
    requirements = [30, 80, 65, 35, 40]
    C = [[8, 12, 4, 9, 10],
         [7, 5, 15, 3, 6],
         [9, 40, 6, 12, 7],
         [5, 30, 2, 6, 4]]

    # stock = [160, 140, 170]
    # requirements = [120, 50, 190, 110]
    # C = [
    #         [7, 8, 1, 2],
    #         [4, 5, 9, 8],
    #         [9, 2, 3, 6],
    #  ]

    # stock = [100, 150, 200, 100]
    # requirements = [120, 200, 100, 30, 100]
    # C = [[7, 5, 6, 9, 5],
    #      [4, 5, 8, 8, 10],
    #      [3, 2, 5, 4, 4],
    #      [9, 11, 10, 8, 11]]

    check_stock = stock.copy()
    check_requirements = requirements.copy()
    check_cost = C.copy()

    m = len(C)
    n = len(C[0])
    X = [[-1 for k in range(len(C[0]))] for n in range(len(C))]
    X = np.array(X)

    equality()
    X = north_west_corner_method()
    indexes = get_vertexes_for_dfs()
    method_of_potencials(indexes)

    print('\n\n## Check my program for optimization using `pulp` ##')
    show_right_answer(supply=check_stock, demand=check_requirements, cost=check_cost)
