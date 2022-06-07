from graphcoloring import GraphGenerator

mat = GraphGenerator.create_rand_adj_mat(4)
print(mat)
GraphGenerator.write_adj_list(mat)