import torch


def get_dataset(file_path):
    data_tensor = torch.load(file_path)
    # print(data_tensor.cuda())
    return data_tensor


XY = get_dataset("production/VNUM_4_DATA_ARRAY.pt")
X = XY[0]
Y = XY[1]
is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()
is_input = is_input.view(is_input.size(0), -1)
is_input = is_input.cuda()
print(is_input[4095])
