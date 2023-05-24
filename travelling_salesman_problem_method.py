import numpy as np
from copy import deepcopy
from python_tsp.exact import solve_tsp_dynamic_programming

def get_di(A):
     return [min(A[i]) if min(A[i]) != np.inf else 0 for i in range(len(A))]

def subtract_di(A, d_i_list):
     for i in range(len(A)):
          for j in range(len(A[0])):
               if A[i][j] != np.inf:
                    A[i][j] -= d_i_list[i]
     return A

def get_columns(A):
     return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def get_dj(A):
     columns = get_columns(A)
     return [min(columns[i]) if min(columns[i]) != np.inf else 0 for i in range(len(A))]

def transport_matrix(matrix):
     rows = len(matrix)
     cols = len(matrix[0])
     transposed_matrix = [[0 for _ in range(rows)] for _ in range(cols)]
     for i in range(rows):
          for j in range(cols):
               transposed_matrix[j][i] = matrix[i][j]
     return transposed_matrix

def subtract_dj(A, d_j_list):
     columns = get_columns(A)
     for i in range(len(columns)):
          for j in range(len(columns[0])):
               if columns[i][j] != np.inf:
                    columns[i][j] -= d_j_list[i]
     A = transport_matrix(columns)
     return A


def reduct_table(A_init):
     A = deepcopy(A_init)
     d_i = get_di(A)
     A = subtract_di(A, d_i)
     d_j = get_dj(A)
     A = subtract_dj(A, d_j)
     H = sum(d_i) + sum(d_j)

     return A, H


def get_edge(A):
     di = [None] * len(A)
     dj = [None] * len(A[0])
     columns = get_columns(A)

#fill in d_i and d_j
     for i in range(len(A)):
          for j in range(len(A[0])):
               if A[i][j] == 0:
                    A[i][j] = np.inf
                    columns[j][i] = np.inf

                    di[i] = min(A[i])
                    dj[j] = min(columns[j])

                    A[i][j] = 0
                    columns[j][i] = 0

#find edge
     ratios = [[-1 for _ in range(len(A[0]))] for _ in range(len(A))]
     for i in range(len(A)):
          for j in range(len(A[0])):
               if A[i][j] == 0:
                    ratios[i][j] = di[i] + dj[j]

# find idx of max element
     max_val = -np.inf
     max_row, max_col = None, None

     for i in range(len(ratios)):
          for j in range(len(ratios[i])):
               if ratios[i][j] > max_val:
                    max_val = ratios[i][j]
                    max_row, max_col = i, j

     return max_row, max_col

def update_table_with_edge(A, max_row, max_col):
     A = deepcopy(A)
     A[max_col][max_row] = np.inf
     for i in range(len(A)):
          for j in range(len(A)):
               if i == max_row:
                    A[i][j] = np.inf
               if j == max_col:
                    A[i][j] = np.inf

     A, H = reduct_table(A)
     return A, H


def update_table_without_edge(A, max_row, max_col):
     A = deepcopy(A)
     A[max_row][max_col] = np.inf
     A, H = reduct_table(A)

     return A, H

def get_cost(costs, path):
     sum = 0
     for edge in path:
          sum += costs[edge[0]][edge[1]]

     return sum

def check_result(table):
   print('\n## Check results with networkx ##')
   import numpy as np
   import networkx as nx

   # Create a complete graph with the given distances as edge weights
   G = nx.DiGraph(np.array(table))
   tour = nx.algorithms.approximation.traveling_salesman.held_karp_ascent(G)
   print("Optimal cost:", tour[0])
   print("Optimal tour:", tour[1].edges)




def main():
     #my option
     # costs = [
     #    [np.inf, 3, 18, 9, 19],
     #    [18, np.inf, 20, 9, 2],
     #    [15, 19, np.inf, 15, 17],
     #    [18, 19, 7, np.inf, 4],
     #    [2, 3, 9, 17, np.inf],
     # ]

     costs = [
          [np.inf, 4, 15, 13, 3],
          [19, np.inf, 20, 18, 3],
          [70, 17, np.inf, 13, 9],
          [15, 2, 16, np.inf, 9],
          [40, 14, 16, 11, np.inf],
     ]

     A, H = reduct_table(costs)
     nodes = [(A.copy(), H, [])]
     best_cost = np.inf
     best_path = []

     while nodes:
          min_node_idx = 0
          for i in range(len(nodes)):
               if nodes[i][1] < nodes[min_node_idx][1]:
                    min_node_idx = i

          node = nodes[min_node_idx]
          nodes.remove(node)

          A, H, path = node

          if H >= best_cost:
               continue

          if len(path) == len(costs):
               cost = get_cost(costs, path)

               if cost < best_cost:
                    best_cost = cost
                    best_path = path

          else:
               max_row, max_col = get_edge(A)
               if A[max_row][max_col] == np.inf:
                    continue

               A_with_edge, H_with_edge = update_table_with_edge(A, max_row, max_col)

               A_without_edge, H_without_edge = update_table_without_edge(A, max_row, max_col)

               H_with_edge = H + H_with_edge
               H_without_edge = - H_without_edge + H

               if H_with_edge < best_cost:
                   nodes.append((A_with_edge, H_with_edge, path+[(max_row, max_col)]))
               if H_without_edge < best_cost:
                   nodes.append((A_without_edge, H_without_edge, path))

     print(f'Cost:{best_cost}')
     print(f'Path:{best_path}')

     check_result(costs)


main()
