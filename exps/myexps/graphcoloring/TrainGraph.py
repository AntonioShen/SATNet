import torch


def get_dataset(file_path):
    data_tensor = torch.load(file_path)
    print(data_tensor)
    print(data_tensor.cuda())


get_dataset("production/VNUM_10_DATA_ARRAY.pt")
