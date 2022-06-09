from ctypes import *
from GraphGenerator import create_rand_adj_mat
from GraphGenerator import write_overheads_adj_list
from tqdm import tqdm
import numpy as np


def mapping_by_coloring(max_color_count, mat, coloring_path="production/coloring.csv", flattening=True):
    mapped_mat = np.zeros((mat.shape[0], mat.shape[1], max_color_count))
    coloring_solution = open(coloring_path, "r")
    coloring_arr = np.fromstring(coloring_solution.readline(), sep=",") - 1
    coloring_solution.close()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 1:
                temp = mapped_mat[i][j]
                temp[int(coloring_arr[i])] = 1
                mapped_mat[i][j] = temp
    if flattening:
        return mapped_mat.flatten()
    else:
        return mapped_mat


def data_pipeline(v_num, padding=False):
    quantity = 2 ** (v_num * v_num - v_num)
    if quantity > 6000:
        quantity = 6000
    dataset = 0
    so_file = "UndirectedVertexColoringSolver.so"
    resolve_coloring = CDLL(so_file)
    if not padding:
        mat_size = v_num
    print("Creating graph dataset: " + str(quantity) + " graphs")
    for i in tqdm(range(quantity)):
        mat = create_rand_adj_mat(size=mat_size)
        write_overheads_adj_list(mat)
        resolve_coloring.solution()
        if i == 0:
            dataset = mapping_by_coloring(max_color_count=v_num, mat=mat)
        else:
            dataset = np.vstack((dataset, mapping_by_coloring(max_color_count=v_num, mat=mat)))
    print(dataset.shape)
    return dataset


v_num = 4
dataset = data_pipeline(v_num)
print("Saving dataset to production/VNUM_" + str(v_num) + "_DATA_ARRAY.npy")
with open("production/VNUM_" + str(v_num) + "_DATA_ARRAY.npy", "wb") as f:
    np.save(f, np.array(dataset))
print("Saved")
