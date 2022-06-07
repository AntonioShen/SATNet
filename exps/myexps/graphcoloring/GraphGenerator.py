import numpy as np

def create_rand_adj_mat(size):
    mat = np.random.rand(size, size)
    # for col in range(size):
    #     col_ = col + 1
    #     for i in range(size):
    #         for j in range(size):
    #             if i == j:
    #                 mat[i, j] = 0
    #             else:
    #                 if mat[i, j] >= 0.5:
    #                     mat[i, j] = 1
    #                 else:
    #                     mat[i, j] = 0
    #                 mat[j, i] = mat[i, j]
    j_lim = 1
    for i in range(size):
        for j in range(j_lim):
            if i == j:
                mat[i, j] = 0
            else:
                if mat[i, j] >= 0.5:
                    mat[i, j] = 1
                else:
                    mat[i, j] = 0
                mat[j, i] = mat[i, j]
        j_lim = j_lim + 1
    return mat

def write_adj_list(mat):
    #file = open("adj_list.csv", "a")
    mat = np.array(mat)
    v_num = mat.shape[0]
    e_num = 0
    j_lim = 1
    for i in range(v_num):
        for j in range(j_lim):
            if mat[i][j] == 1:
                e_num = e_num + 1
        j_lim = j_lim + 1
    print(v_num, e_num)


    #file.close()
    return 0