from ctypes import *
from GraphGenerator import create_rand_adj_mat
from GraphGenerator import write_overheads_adj_list
from tqdm import tqdm
import numpy as np


def mapping_by_coloring(max_color_count, mat, coloring_path="production/coloring.csv", flattening=True, encoder="normal"):
    # 1 maps to [0, 1, ..., 0], index 0 is reserved.
    mapped_mat = np.zeros((mat.shape[0], mat.shape[1], (max_color_count + 1)))
    coloring_solution = open(coloring_path, "r")
    coloring_arr = np.fromstring(coloring_solution.readline(), sep=",")
    coloring_solution.close()
    if encoder == "normal":
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i][j] == 1:
                    temp = mapped_mat[i][j]
                    temp[int(coloring_arr[i])] = 1
                    mapped_mat[i][j] = temp
    elif encoder == "reversed":
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                temp = mapped_mat[i][j]
                if mat[i][j] == 0:
                    temp[int(coloring_arr[i])] = 1
                    mapped_mat[i][j] = temp
                else:
                    temp[0] = 1
                    mapped_mat[i][j] = temp
    if flattening:
        return mapped_mat.flatten()
    else:
        return mapped_mat


def to_catalog_2d(mat, offset=0, catalog=0, flattening=True):   # Catalog number starts with 1.
    mat = np.array(mat)
    if catalog == 0:
        catalog = np.amax(mat) + offset
    temp = np.zeros((mat.shape[0], mat.shape[1], catalog))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] != 0:
                temp[i][j][int(mat[i][j] - 1 + offset)] = 1
    if flattening:
        return temp.flatten()
    else:
        return temp


def data_pipeline(v_num, padding=False, encoder="normal"):
    quantity = 2 ** (v_num * v_num - v_num)
    catalog = v_num + 1
    if quantity > 6000:
        quantity = 6000
    data = 0
    label = 0
    so_file = "UndirectedVertexColoringSolver.so"
    resolve_coloring = CDLL(so_file)
    if not padding:
        mat_size = v_num
    print("Creating graph dataset: " + str(quantity) + " graphs " + encoder)
    if encoder == "normal":
        for i in tqdm(range(quantity)):
            mat = create_rand_adj_mat(size=mat_size)
            write_overheads_adj_list(mat)
            resolve_coloring.solution()
            if i == 0:
                data = to_catalog_2d(mat, catalog=catalog)
                label = mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder)
            else:
                data = np.vstack((data, to_catalog_2d(mat, catalog=catalog)))
                label = np.vstack((label, mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder)))
    elif encoder == "reversed":
        for i in tqdm(range(quantity)):
            mat = create_rand_adj_mat(size=mat_size)
            write_overheads_adj_list(mat)
            resolve_coloring.solution()
            mat = np.array(mat)
            mat[mat == 0] = 2
            mat[mat == 1] = 0
            mat[mat == 2] = 1
            if i == 0:
                data = to_catalog_2d(mat, catalog=catalog)
                label = mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder)
            else:
                data = np.vstack((data, to_catalog_2d(mat, catalog=catalog)))
                label = np.vstack((label, mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder)))
    print(data.shape)
    print(label.shape)
    return data, label


v_num = 2
dataset, label = data_pipeline(v_num, encoder="reversed")
data = np.vstack((np.reshape(dataset, (1, dataset.shape[0], dataset.shape[1])),
                  np.reshape(label, (1, label.shape[0], label.shape[1]))))
print(data)
print("Saving dataset to production/VNUM_" + str(v_num) + "_DATA_ARRAY.npy")
with open("production/VNUM_" + str(v_num) + "_DATA_ARRAY.npy", "wb") as f:
    np.save(f, data)
print("Saved")
with open("production/VNUM_" + str(v_num) + "_DATA_ARRAY.npy", "rb") as f:
    a = np.load(f)
