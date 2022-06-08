from graphcoloring import GraphGenerator

mat = GraphGenerator.create_rand_adj_mat(4)
print(mat)
GraphGenerator.write_overheads_adj_list(mat)
