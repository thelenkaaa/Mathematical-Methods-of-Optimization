import numpy as np

M = 100000

def multiply_columns(col1, col2):
    result_column = []
    for i in range(len(col1)):
        res = col1[i]*col2[i]
        result_column.append(res)
    return result_column

def column_sum(col):
    result = 0
    for i in range(len(col)):
        result += col[i]
    return result

def get_leading_row(P, A, leading_column):
    count = 0
    for i in range(len(A)):
        if A[i][leading_column] < 0:
            count += 1
    if count == len(A):
        print("Solution can not be optimized!")
    grades = []
    for i in range(len(A)):
        if A[i][leading_column] <= 0:
            grades.append(0)
        else:
            grades.append(P[i]/A[i][leading_column])
    if min_positive_number(grades) == None:
        return None
    else:
        leading_row = grades.index(min_positive_number(grades))
        return leading_row

def updated_leading_row(prev_elements, cross_num, P, ):
    new_plan = P/cross_num
    return new_plan, [prev_elements[i]/cross_num for i in range(len(prev_elements))]

def is_unit_column(row):
    return sum(row) == 1 and all(elem in [0, 1] for elem in row)

def basis(A, c):
    basis_list = [-1]*len(A)
    num_cols = len(A[0])

    for j in range(num_cols):

        column = []
        for i in range(len(A)):
            column.append(A[i][j])
        if is_unit_column(column):
            pos = column.index(1)

            basis_list[pos] = j

    if -1 in basis_list:
        c.append(M)
        for i in range(len(A)):
            A[i].append(0)
        A[basis_list.index(-1)][-1] = 1
        basis_list[basis_list.index(-1)] = len(c) - 1

    return basis_list

def column(A):
    columns = []
    for j in range(len(A[0])):
        column = []
        for i in range(len(A)):
            column.append(A[i][j])
        columns.append(column)
    return columns

def min_positive_number(numbers):
    positive_numbers = [n for n in numbers if n > 0]
    if not positive_numbers:
        return None
    else:
        return min(positive_numbers)

def update_row(A, leading_row, leading_column, i, plan):
    new_plan = plan[i] - plan[leading_row] * A[i][leading_column] / A[leading_row][leading_column]
    return new_plan, [A[i][j] - A[leading_row][j] * A[i][leading_column] / A[leading_row][leading_column] for j in range(len(A[i]))]

def is_optimal(m):
    for i in m:
        if i > 0:
            return False
    return True

# функція з 1 лаби
# A = [[5, -2, 1, 0, 0],
#      [-1, 2, 0, 1, 0],
#      [1, 1, 0, 0, -1]]
# c = [-3, 6, 0, 0, 0]
# P = [4, 4, 4]

# random
# A = [[1, -6, 1, 0, 0],
#      [13, 3, 0, -1, 0],
#      [3, 7, 0, 1, 1]]
# P = [1, 6, 30]
# c = [2, 4, 0, 0, 0]

# функція з 2 лаби
# A = [[2, 3, 1, 0, 0],
#    [3, 1, 0, 1, 0],
#    [-2, 5, 0, 0, 1]]
# P = [17, 15, 15]
# c = [-2, -3, 0, 0, 0]

# з 2 штучними базисами
# A = [[8, -5, 1, 0, 1],
#      [1, 3, 0, -1, 0],
#      [2, -7, 3, 0, 1]]
# P = [16, 2, 9]
# c = [10, 39, 0, 0, 0]

#мій
# A = [[3, 1, 1, 0, 0, 0, 0],
#      [3, 1, 0, -1, 0, 0, 0],
#      [1, 8, 0, 0, -1, 0, 0],
#      [1, 0, 0, 0, 0, 1, 0],
#      [0, 1, 0, 0, 0, 0, 1]]
# P = [4, 4, 4, 3, 3]
# c = [-3, -5, 0, 0, 0, 0, 0]

# A = [[0, 0, 1, 1, 0, 0, 0],
#    [1, 0, 0, -0.34782608, 0.04347826, 0, 0],
#    [0, 1, 0, 0.04347826, -0.13043478, 0, 0],
#    [0, 0, 0, 0.34782608, -0.04347826, 1, 0],
#    [0, 0, 0, -0.04347826, 0.13043478, 0, 1]]
# P = [0, 1.2173913, 0.34782609, 1.7826087, 2.65217391]
# c = [-3, -5, 0, 0, 0, 0, 0]





def simplex_method(A, P, c):
    while True:
        indexes_for_basis = basis(A, c)

        C_b = [c[indexes_for_basis[j]] for j in range(len(indexes_for_basis))]

        # print([f"x{indexes_for_basis[i]+1} = {P[i]}" for i in range(len(P))])

        sum_plan = column_sum(multiply_columns(C_b, P))

        # print("Function:", sum_plan)
        m_summ_of_each_column = []
        columns_of_x_i = column(A)

        for i in range(len(A[0])):
            # checking of m+1 for basis values to be 0
            if i in indexes_for_basis:
                temp = 0
            else:
                temp = column_sum(multiply_columns(columns_of_x_i[i], C_b)) - c[i]
            m_summ_of_each_column.append(temp)

        if is_optimal(m_summ_of_each_column):
            break

        # print(f"Δj:{m_summ_of_each_column}")

        leading_column = m_summ_of_each_column.index(max(m_summ_of_each_column))

        leading_row = get_leading_row(P, A, leading_column)

        koef = A[leading_row][leading_column]

        for i in range(len(A)):
            if i == leading_row:
                P[i], A[leading_row] = updated_leading_row(A[leading_row], koef, P[i])
            else:
                P[i], A[i] = update_row(A, leading_row, leading_column, i, P)

        # print(np.array(A))

    indexes_not_in_basis = list(set(range(len(c))) - set(indexes_for_basis))

    return [p for i, p in sorted(list(zip(indexes_for_basis, P)) + list(zip(indexes_not_in_basis, [0] * len(indexes_not_in_basis))))]


# print("res: ", simplex_method(A, P, c))
# print()
# print("## scipy.optimize.linprog(method='simplex') ##")
# from scipy import optimize
# print(optimize.linprog(c=c, A_eq=A, b_eq=P, method='simplex'))



