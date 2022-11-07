import random
from ctypes import *
from GraphGenerator import create_rand_adj_mat
from GraphGenerator import write_overheads_adj_list
from tqdm import tqdm
import numpy as np
import torch


def mapping_by_coloring(max_color_count, mat, coloring_path="production/coloring.csv", flattening=True,
                        encoder="normal"):
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


def to_catalog_2d(mat, offset=0, catalog=0, flattening=True):  # Catalog number starts with 1.
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


def remove_coloring_from_label(label_mat, ratio, flattening, method='row'):
    removed = np.zeros(label_mat.shape)
    remain_one = np.zeros(label_mat.shape)
    colored_entry = []
    checked = False
    removed_rows = random.choices(range(0, label_mat.shape[0]), k=int((1 - ratio) * label_mat.shape[0]))
    for i in range(label_mat.shape[0]):
        if method == 'row' and (i in removed_rows):
            remove_this_row = True
        else:
            remove_this_row = False
        for j in range(label_mat.shape[1]):
            if label_mat[i][j][0] == 1:
                removed[i][j] = label_mat[i][j]
                remain_one[i][j] = label_mat[i][j]
                continue
            else:
                if not remove_this_row:
                    removed[i][j] = label_mat[i][j]
                if not checked:
                    temp = np.zeros((remain_one.shape[2], ))
                    temp[1] = 1.0
                    # remain_one[i][j] = label_mat[i][j]
                    for k in range(remain_one.shape[1]):
                        if remain_one[i][k][0] != 1.0:
                            remain_one[i][k] = temp
                    checked = True
                if ([i, j] in colored_entry) or ([j, i] in colored_entry):
                    continue
                else:
                    colored_entry.append([i, j])
    if method == 'diag':
        removed_indices = random.choices(range(0, len(colored_entry)), k=int((1 - ratio) * len(colored_entry)))
        for index in removed_indices:
            pos = colored_entry[index]
            removed[pos[0], pos[1]] = np.zeros((removed.shape[2], ))
            removed[pos[1], pos[0]] = np.zeros((removed.shape[2],))
    if flattening:
        return removed.flatten(), remain_one.flatten()
    else:
        return removed, remain_one


def data_pipeline(v_num, quantity=0, padding=False, encoder="normal", abs_path=False, flattening=True,
                  partial_coloring_ratio=0.0):
    if partial_coloring_ratio > 1.0 or partial_coloring_ratio < 0.0:
        print('Wrong argument: partial_coloring_ratio [0.0, 1.0].')
        exit()
    if quantity == 0:
        quantity = int(2 ** ((v_num * v_num - v_num) / 2))
    catalog = v_num + 1
    data = 0
    label = 0
    data_all_color_removed = 0
    data_remain_one_colored = 0
    if abs_path:
        so_file = "/home/xingshen/projects/def-six/xingshen/SATNet/exps/myexps/graphcoloring/UndirectedVertexColoringSolver.so"
    else:
        so_file = "/root/autodl-tmp/SATNet/exps/myexps/graphcoloring/UndirectedVertexColoringSolver.so"
    resolve_coloring = CDLL(so_file)
    if not padding:
        mat_size = v_num
    print("Creating graph dataset: " + str(quantity) + " graphs " + encoder + ", " + str(v_num) + " vertices, " +
          str(partial_coloring_ratio) + " partially colored.")
    # A dictionary to remove identical inputs.
    dict = {}
    if encoder == "normal":
        for i in tqdm(range(quantity)):
            mat = create_rand_adj_mat(size=mat_size)
            while str(mat) in dict:
                mat = create_rand_adj_mat(size=mat_size)
            dict[str(mat)] = i
            write_overheads_adj_list(mat)
            resolve_coloring.solution()
            if i == 0:
                data = to_catalog_2d(mat, catalog=catalog, flattening=flattening)
                label = mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder, flattening=flattening)
            else:
                data = np.vstack((data, to_catalog_2d(mat, catalog=catalog, flattening=flattening)))
                label = np.vstack((label, mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder,
                                                              flattening=flattening)))
    elif encoder == "reversed":
        for i in tqdm(range(quantity)):
            mat = create_rand_adj_mat(size=mat_size)
            while str(mat) in dict:
                mat = create_rand_adj_mat(size=mat_size)
            dict[str(mat)] = i
            write_overheads_adj_list(mat)
            resolve_coloring.solution()
            mat = np.array(mat)
            # Reverse 0s and 1s
            mat[mat == 0] = 2
            mat[mat == 1] = 0
            mat[mat == 2] = 1
            if partial_coloring_ratio == 0.0:
                if i == 0:
                    data_all_color_removed = to_catalog_2d(mat, catalog=catalog, flattening=flattening)
                    label = mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder, flattening=flattening)
                else:
                    data_all_color_removed = np.vstack(
                        (data_all_color_removed, to_catalog_2d(mat, catalog=catalog, flattening=flattening)))
                    label = np.vstack((label, mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder,
                                                                  flattening=flattening)))
            else:
                if i == 0:
                    data_all_color_removed = to_catalog_2d(mat, catalog=catalog, flattening=flattening)
                    label = mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder, flattening=False)
                    data, data_remain_one_colored = remove_coloring_from_label(label, ratio=partial_coloring_ratio,
                                                                               flattening=True)
                    label = label.flatten()
                else:
                    data_all_color_removed = np.vstack(
                        (data_all_color_removed, to_catalog_2d(mat, catalog=catalog, flattening=flattening)))
                    temp_label = mapping_by_coloring(max_color_count=v_num, mat=mat, encoder=encoder, flattening=False)
                    temp_data, temp_data_remain_one_colored = remove_coloring_from_label(temp_label,
                                                                                         ratio=partial_coloring_ratio,
                                                                                         flattening=True)
                    data = np.vstack((data, temp_data))
                    data_remain_one_colored = np.vstack((data_remain_one_colored, temp_data_remain_one_colored))
                    label = np.vstack((label, temp_label.flatten()))

    print(data.shape)
    print(label.shape)
    print(data_all_color_removed.shape)
    print(data_remain_one_colored.shape)
    return data, data_all_color_removed, data_remain_one_colored, label


v_num = 20
pr = 0.4
quant = 100000

dataset, dataset_all_color_removed, dataset_remain_one_colored, label = data_pipeline(v_num, quantity=quant,
                                                                                      encoder="reversed",
                                                                                      partial_coloring_ratio=pr)  #
# Flattening=True
data = np.vstack((np.reshape(dataset, (1, dataset.shape[0], v_num, v_num, v_num + 1)),
                  np.reshape(label, (1, label.shape[0], v_num, v_num, v_num + 1))))
data_all_color_removed = np.vstack(
    (np.reshape(dataset_all_color_removed, (1, dataset_all_color_removed.shape[0], v_num, v_num, v_num + 1)),
     np.reshape(label, (1, label.shape[0], v_num, v_num, v_num + 1))))
data_remain_one_colored = np.vstack(
    (np.reshape(dataset_remain_one_colored, (1, dataset_remain_one_colored.shape[0], v_num, v_num, v_num + 1)),
     np.reshape(label, (1, label.shape[0], v_num, v_num, v_num + 1))))
print(data.shape)
print(data_all_color_removed.shape)
print(data_remain_one_colored.shape)
data_tensor = torch.from_numpy(data)
data_all_color_removed_tensor = torch.from_numpy(data_all_color_removed)
data_remain_one_colored_tensor = torch.from_numpy(data_remain_one_colored)
# Shape: (2, :, v_num, v_num, v_num + 1)
print("Saving dataset to production/VNUM_" + str(v_num) + "_DATA_ARRAY_PAR_" + str(pr) + "_QTY_" + str(quant) + ".pt")
torch.save(data_tensor, "production/VNUM_" + str(v_num) + "_DATA_ARRAY_PAR_" + str(pr) + "_QTY_" + str(quant) + ".pt")
print("Saved")
print("Saving dataset to production/VNUM_" + str(v_num) + "_DATA_ARRAY_PAR_" + str(pr) + "_RO_QTY_" + str(quant) + ".pt")
torch.save(data_remain_one_colored_tensor, "production/VNUM_" + str(v_num) + "_DATA_ARRAY_PAR_" + str(pr) + "_RO_QTY_" + str(quant) + ".pt")
print("Saved")
print("Saving dataset to production/VNUM_" + str(v_num) + "_DATA_ARRAY_PAR_" + str(pr) + "_AR_QTY_" + str(quant) + ".pt")
torch.save(data_all_color_removed_tensor, "production/VNUM_" + str(v_num) + "_DATA_ARRAY_PAR_" + str(pr) + "_AR_QTY_" + str(quant) + ".pt")
print("Saved")
