import numpy as np

M = 100000000

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

def updated_leading_row(prev_elements, cross_num, P):
    new_plan = P/cross_num
    return round(new_plan, 8), [round((prev_elements[i]/cross_num), 8) for i in range(len(prev_elements))]

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
    if len(positive_numbers) == 0:
        print("There are no positive values!")
        return None
    else:
        return min(positive_numbers)

def update_row(A, leading_row, leading_column, i, plan):
    new_plan = plan[i] - plan[leading_row] * A[i][leading_column] / A[leading_row][leading_column]
    return round(new_plan, 8), [round((A[i][j] - A[leading_row][j] * A[i][leading_column] / A[leading_row][leading_column]), 8) for j in range(len(A[i]))]

def is_optimal(m):
    for i in m:
        if i > 0:
            return False
    return True

def simplex_method(A, P, c):
    iteration = 0

    while True:

        if iteration > 0:
            print(f'\nIteration:{iteration}')
        indexes_for_basis = basis(A, c)

        C_b = [c[indexes_for_basis[j]] for j in range(len(indexes_for_basis))]

        print([f"x{indexes_for_basis[i] + 1} = {P[i]}" for i in range(len(P))])

        sum_plan = column_sum(multiply_columns(C_b, P))

        print("Function:", sum_plan)
        print(np.array(A))
        m_summ_of_each_column = []
        columns_of_x_i = column(A)

        for i in range(len(A[0])):
            # checking of m+1 for basis values to be 0
            if i in indexes_for_basis:
                temp = 0
            else:
                temp = column_sum(multiply_columns(columns_of_x_i[i], C_b)) - c[i]
            m_summ_of_each_column.append(temp)

        print(f"Δj:{m_summ_of_each_column}")

        if is_optimal(m_summ_of_each_column):
            break

        leading_column = m_summ_of_each_column.index(max(m_summ_of_each_column))

        if get_leading_row(P, A, leading_column) == None:
            break
        else:
            leading_row = get_leading_row(P, A, leading_column)

        koef = A[leading_row][leading_column]

        for i in range(len(A)):
            if i == leading_row:
                P[i], A[leading_row] = updated_leading_row(A[leading_row], koef, P[i])
            else:
                P[i], A[i] = update_row(A, leading_row, leading_column, i, P)

        iteration += 1


# functions for dual method
def dual_get_leading_row(P):
    return P.index(min(P))

#not sure about thissssssss
def dual_get_leading_col(m, row):
    min_val = 1000
    pos = -1
    for j in range(len(m)):
        #checking for devision by zero and basis grades(0)
        if row[j] < 0 and m[j] != 0:
            val = abs(m[j]/row[j])
            if val < min_val:
                min_val = val
                pos = m.index(m[j])
    if pos == -1:
        return None
    return pos

def dual_check_plan(P):
    count = 0
    for i in range(len(P)):
        if P[i] < 0:
            count += 1
    if count == 0:
        print("No negative values in plan left")
        return False
    return True

def dual_check_leading_row(row):
    count = 0
    for i in range(len(row)):
        if row[i] < 0:
            count += 1
    if count == 0:
        print("Function is unlimited!")
        return False
    return True


def dual_method(A, P, c):
    print('Solution is solved with dual method')
    iteration = 1

    while True:
        print(f"\nIteration {iteration}")

        leading_row = dual_get_leading_row(P)

        # checking elements of leading row(at least 1 must be negative)
        if not dual_check_leading_row(A[leading_row]):
            break

        indexes_for_basis = basis(A, c)

        C_b = [c[indexes_for_basis[j]] for j in range(len(indexes_for_basis))]

        print([f"x{indexes_for_basis[i] + 1} = {P[i]}" for i in range(len(P))])

        sum_plan = column_sum(multiply_columns(C_b, P))

        print("Function:", sum_plan)
        print(np.array(A))

        m_summ_of_each_column = []
        columns_of_x_i = column(A)

        for i in range(len(A[0])):
            # checking of m+1 for basis values to be 0
            if i in indexes_for_basis:
                temp = 0
            else:
                temp = column_sum(multiply_columns(columns_of_x_i[i], C_b)) - c[i]
            m_summ_of_each_column.append(temp)

        print(f"Δj:{m_summ_of_each_column}")

        leading_column = dual_get_leading_col(m_summ_of_each_column, A[leading_row])
        if leading_column == None:
            print("Can't choose leading column")
            break
        leading_element = A[leading_row][leading_column]

        for i in range(len(A)):
            if i == leading_row:
                P[i], A[leading_row] = updated_leading_row(A[leading_row], leading_element, P[i])
            else:
                P[i], A[i] = update_row(A, leading_row, leading_column, i, P)
        iteration += 1

        if not dual_check_plan(P):
            break

# A = [[1, 1, 1, 0, 0, 0, 0],
#      [-3, -1, 0, 1, 0, 0, 0],
#      [-1, -5, 0, 0, 1, 0, 0],
#      [1, 0, 0, 0, 0, 1, 0],
#      [0, 1, 0, 0, 0, 0, 1]]
# P = [4, -4, -4, 3, 3]
# c = [3, -3, 0, 0, 0, 0, 0]

c = [-21, -30, -16, 0, 0]
A = [
    [-3, -2, -3, 1, 0],
    [-1, -3, 0, 0, 1]
]
P = [3, -2]


def dual_primal(P):
    count = 0
    for i in range(len(P)):
        if P[i] < 0:
            count += 1
    if count > 0:
        return True
    return False

if dual_primal(P) == False:
    print("Dual method does not apply here!")
    simplex_method(A, P, c)
else:
    dual_method(A, P, c)
    simplex_method(A, P, c)



print()
print("## scipy.optimize.linprog(method='simplex') ##")
from scipy import optimize
print(optimize.linprog(c=c, A_eq=A, b_eq=P, method='simplex'))



# lecture example
# A = [[-3, -1, 1, 0, 0],
#      [-4, -3, 0, 1, 0],
#      [1, 2, 0, 0, 1]]
# P = [-3, -6, 3]
# c = [2, 1, 0, 0, 0]
# first option
# A = [[2, 3, 1, 0, 0],
#      [2, 1, 0, 1, 0],
#      [-1, -1, 0, 0, 1]]
# P = [8, 5, -1]
# c = [-1, -2, 0, 0, 0]
# my option
# A = [[1, 1, 1, 0, 0, 0, 0],
#      [-3, -1, 0, 1, 0, 0, 0],
#      [-1, -5, 0, 0, 1, 0, 0],
#      [1, 0, 0, 0, 0, 1, 0],
#      [0, 1, 0, 0, 0, 0, 1]]
# P = [4, -4, -4, 3, 3]
# c = [-3, -3, 0, 0, 0, 0, 0]