import numpy as np

def create_rand_adj_mat(size):
    mat = np.random.rand(size, size)
    for col in range(size):
        col_ = col + 1
        for i in range(col_):
            for j in range(size):
                if i == j:
                    mat[i, j] = 0
                else:
                    if mat[i, j] >= 0.5:
                        mat[i, j] = 1
                    else:
                        mat[i, j] = 0
                    mat[j, i] = mat[i, j]
    return mat