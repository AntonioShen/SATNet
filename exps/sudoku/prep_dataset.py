from pathlib import Path
from collections import namedtuple

from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets
from tqdm.auto import tqdm
from sudoku import Sudoku

import matplotlib.pyplot as plt
import numpy as np

import torch
import random

# We fix this value, unlike the original code, since it cannot drop below 0.1 without MNIST 
# image repeats between train and test data.
TEST_PCT = 0.1 

def _to_visual(X, is_input, mnist_x, mnist_y):

    # mean = torch.mean(mnist_x)
    # std = torch.std(mnist_x)

    # mnist_x = mnist_x*2
    # mnist_x = torch.clamp(mnist_x, 0, 1)

    assert X[~is_input.type(torch.uint8)].sum() == 0

    X = X.view(-1, 9)

    mnist_x = mnist_x.repeat(10, 1, 1, 1)
    mnist_y = mnist_y.repeat(10)

    out = torch.zeros((X.shape[0], 1, 28, 28), requires_grad=False)
    for i in range(9):
        Y_curr = X[:, i] == 1
        mnist_curr = mnist_y == (i + 1)

        out[Y_curr] = mnist_x[mnist_curr][:torch.sum(Y_curr.int())]

    out = out.view(-1, 81, 1, 28, 28).contiguous()

    return out

def _get_infogan_labels_optional(label_dir):
    infogan_labels = None
    if label_dir is not None:
        label_dir = Path(label_dir)
        with open(label_dir/'train_labels.pt', 'rb') as f:
            labels_train = torch.load(f)
        with open(label_dir/'test_labels.pt', 'rb') as f:
            labels_test = torch.load(f)

        infogan_labels = torch.cat([labels_train, labels_test])

    return infogan_labels



    def generate_data_partition(num_data, mnist, numerical_perm):

        outputs_numerical = torch.zeros(num_data, 20, dtype=torch.long)
        outputs_numerical[:, :10] = torch.randint(9, (num_data, 10))

        outputs_numerical[:, 10:] = outputs_numerical[:, :10][:, numerical_perm]

        # to one hot version
        outputs = torch.zeros(num_data, 20, 9)
        for i in range(20):
            outputs[:, i].scatter_(1, outputs_numerical[:, i].view(-1, 1), 1)

        inputs = torch.zeros(num_data, 20, 1, 28, 28)
        for position_index in range(10):
            for value_index in range(9):
                source_indices = mnist.targets == value_index + 1
                destination_indices = outputs_numerical[:, position_index] == value_index
                inputs[destination_indices, position_index, 0] = mnist.data[source_indices, 0][:destination_indices.int().sum()]

            # Shuffle for different draws of the same digit per row.
            perm = torch.randperm(mnist.data.shape[0])
            mnist.data = mnist.data[perm]
            mnist.targets = mnist.targets[perm]

        mask = torch.zeros(num_data, 20, 9, dtype=torch.int32)
        mask[:, :10, :] = 1

        return inputs, mask, outputs

    args = generate_data_partition(int(1e4), mnist_train, numerical_perm)

    grid = torchvision.utils.make_grid(args[0][0], nrow=20)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig('input_grid.png')

    # if infogan_labels_train is not None: args.append(infogan_labels_train)
    if cuda: args = (arg.cuda() for arg in args)
    train_dataset = TensorDataset(*args)

    args = generate_data_partition(int(1e3), mnist_test, numerical_perm)
    # if infogan_labels_train is not None: args.append(infogan_labels_train)
    if cuda: args = (arg.cuda() for arg in args)
    test_dataset = TensorDataset(*args)


    return train_dataset, test_dataset


def get_sudoku_visual(data_dir, cuda, infogan_label_dir=None):
    
    X, Ximg, Y, is_input = _get_sudoku_core_initialize(data_dir)

    infogan_labels = _get_infogan_labels_optional(infogan_label_dir)

    return _get_sudoku_core_datasets(Ximg, is_input, Y, cuda, infogan_labels=infogan_labels)

def get_sudoku_nonvisual(data_dir, cuda, perm, num_injected_input_cell_errors=0, solvability='any'):

    X, Ximg, Y, is_input = _get_sudoku_core_initialize(data_dir)

    # Noisy inputs & solvability bl

    data = _inject_input_cell_errors(X, is_input, num_injected_input_cell_errors, solvability)

    N = data.size(0)
    nTrain = int(N*(1.-TEST_PCT))
    solvable = _get_solvability(data[nTrain:], is_input[nTrain:]) if num_injected_input_cell_errors > 0 else None

    # Perm.
    unperm = None
    if perm is not None:
        print('Applying permutation')
        # zyy
        # data[:,:], Y[:,:], is_input[:,:] = data[:,perm], Y[:,perm], is_input[:,perm]
        Y[:, :] = Y[:,perm]
        unperm = _find_unperm(perm)

    train_set, test_set = _get_sudoku_core_datasets(data, is_input, Y, cuda, solvable)
    return train_set, test_set, unperm

    

def _get_sudoku_core_initialize(data_dir):
    data_dir = Path(data_dir)

    with open(data_dir/'features.pt', 'rb') as f:
        X_in = torch.load(f)
    with open(data_dir/'features_img.pt', 'rb') as f:
        Ximg_in = torch.load(f)
    with open(data_dir/'labels.pt', 'rb') as f:
        Y_in = torch.load(f)
    with open(data_dir/'perm.pt', 'rb') as f:
        perm = torch.load(f)

    X, Ximg, Y, is_input = _process_inputs(X_in, Ximg_in, Y_in, 3)

    return X, Ximg, Y, is_input


def _get_sudoku_core_datasets(data, is_input, Y, cuda, solvable=None, infogan_labels=None):

    N = data.size(0)
    nTrain = int(N*(1.-TEST_PCT))

    args = [data, is_input, Y]
    if infogan_labels is not None: args.append(infogan_labels)

    if cuda: args = [arg.cuda() for arg in args]

    train_set = TensorDataset(*[arg[:nTrain] for arg in args])

    # FIXME
    if solvable is not None:
        test_set = TensorDataset(*[arg[nTrain:] for arg in args], solvable)
    else:
        test_set = TensorDataset(*[arg[nTrain:] for arg in args])


    return train_set, test_set

def _inject_input_cell_errors(data, is_input, num_errors, solvability):

    if num_errors == 0: return data

    print('Injecting Errors...')
    
    data_before = data.clone()
    for i, (x, is_i) in enumerate(tqdm(zip(data, is_input), total=data.shape[0])):
        new_x = _inject_input_cell_errors_single(x.clone(), is_i, num_errors).flatten()

        if solvability == 'unsolvable':
            while _get_solvability_single(new_x, is_i):
                new_x = _inject_input_cell_errors_single(x.clone(), is_i, num_errors).flatten()
        elif solvability == 'solvable':
            while not _get_solvability_single(new_x, is_i):
                new_x = _inject_input_cell_errors_single(x.clone(), is_i, num_errors).flatten()

        data[i] = new_x

    select = is_input > 0
    data_before[~select] = 0

    data_after = data.clone()
    data_after[~select] = 0
    diff = torch.sum(torch.any(data_before != data_after, dim=1).int()).float()
    assert diff == data.shape[0]  # Sanity check. All should contain an erroneous input.

    return data

def _inject_input_cell_errors_single(x, is_i, num_errors):
    x = x.view(-1, 9)
    is_i = is_i.view(-1, 9)

    inputs = is_i[:, 0].nonzero().flatten()
    chosen_indices = np.random.choice(inputs, num_errors, replace=False)


    for index in chosen_indices:
        new_digit = random.randint(0, 8)
        while x[index, new_digit] == 1:
            new_digit = random.randint(0, 8)

        x[index].fill_(0)
        x[index, new_digit] = 1

    return x

def _process_inputs(X, Ximg, Y, boardSz, normalize=True):
    is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()

    Ximg = Ximg.flatten(start_dim=1, end_dim=2)
    Ximg = Ximg.unsqueeze(2).float()

    if normalize:
        mean = torch.mean(Ximg)
        std = torch.std(Ximg)

        Ximg = (Ximg - mean) / std
        Ximg = torch.clamp(Ximg, 0, 1)

    X      = X.view(X.size(0), -1)
    Y      = Y.view(Y.size(0), -1)
    is_input = is_input.view(is_input.size(0), -1)

    return X, Ximg, Y, is_input


def _find_unperm(perm):
    unperm = torch.zeros_like(perm)
    for i in range(perm.size(0)):
        unperm[perm[i]] = i
    return unperm

def _get_solvability(data, is_input):
    print('Obtaining solvability...')

    solvable = []
    for x, is_i in tqdm(zip(data, is_input), total=data.shape[0]):
        solvable.append(_get_solvability_single(x, is_i))

    return torch.ByteTensor(solvable)

def _get_solvability_single(x, is_i):
    is_i = is_i.view(81, 9).float().mean(dim=1).type(torch.uint8).view(9, 9)
    x = x.view(81, 9).argmax(dim=1).view(9, 9)

    x[~is_i] = 0
    puzzle = Sudoku(3, 3, board=x.tolist())

    return puzzle.solve().validate()