# from graphcoloring import GraphGenerator
#
# mat = GraphGenerator.create_rand_adj_mat(4)
# print(mat)
# GraphGenerator.write_overheads_adj_list(mat)
import torch
from torch.utils.cpp_extension import CUDA_HOME


if torch.cuda.is_available() and CUDA_HOME is not None:
    print("It's available.")
