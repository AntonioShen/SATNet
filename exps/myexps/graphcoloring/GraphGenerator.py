import numpy as np


def create_rand_adj_mat(size):
    mat = np.random.rand(size, size)
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


def extract_adj_list(mat):
    mat = np.array(mat)
    adj_list = np.ones((1, 3))
    v_num = mat.shape[0]    # Obtaining overheads.
    e_num = 0
    j_lim = 1
    for i in range(v_num):
        for j in range(j_lim):
            if mat[i][j] == 1:
                adj_list = np.vstack((adj_list, [1, 1, 1]))
                adj_list[e_num] = [i, j, 1]
                e_num = e_num + 1
        j_lim = j_lim + 1
    adj_list = np.delete(adj_list, e_num, 0)
    return v_num, e_num, adj_list


def write_overheads_adj_list(mat, file_path="adj_list.sst"):
    v_num, e_num, adj_list = extract_adj_list(mat)
    file = open(file_path, "a")
    file.writelines(str(v_num) + " " + str(e_num) + '\n')  # Writing overheads.
    adj_list_str = np.array2string(np.array(adj_list), separator=",", threshold=(e_num * 8))
    adj_list_str = adj_list_str.translate({ord(i): None for i in '[] .'})
    adj_list_str = adj_list_str.replace(",", " ")
    file.writelines(adj_list_str + '\n')
    file.close()
    return 0

