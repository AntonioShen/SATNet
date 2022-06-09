from ctypes import *
from GraphGenerator import create_rand_adj_mat
from GraphGenerator import write_overheads_adj_list

so_file = "UndirectedVertexColoringSolver.so"
resolve_coloring = CDLL(so_file)

mat = create_rand_adj_mat(10)
print(mat)
write_overheads_adj_list(mat)
resolve_coloring.solution()
