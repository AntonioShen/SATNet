import torch
from torch.utils.data import TensorDataset
from collections import namedtuple

args_dict = {'lr': 2e-3,
             'cuda': torch.cuda.is_available(),
             'batchSz': 40,
             'mnistBatchSz': 50,
             'boardSz': 3, # for 9x9 Sudoku
             'm': 600,
             'aux': 300,
             'nEpoch': 100
            }
args = namedtuple('Args', args_dict.keys())(*args_dict.values())

def process_inputs(X, Ximg, Y, boardSz):
    is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()

    Ximg = Ximg.flatten(start_dim=1, end_dim=2)
    Ximg = Ximg.unsqueeze(2).float()

    X      = X.view(X.size(0), -1)
    Y      = Y.view(Y.size(0), -1)
    is_input = is_input.view(is_input.size(0), -1)

    return X, Ximg, Y, is_input

with open('sudoku/features.pt', 'rb') as f:
    X_in = torch.load(f)
with open('sudoku/features_img.pt', 'rb') as f:
    Ximg_in = torch.load(f)
with open('sudoku/labels.pt', 'rb') as f:
    Y_in = torch.load(f)
with open('sudoku/perm.pt', 'rb') as f:
    perm = torch.load(f)

X, Ximg, Y, is_input = process_inputs(X_in, Ximg_in, Y_in, args.boardSz)
if args.cuda: X, Ximg, is_input, Y = X.cuda(), Ximg.cuda(), is_input.cuda(), Y.cuda()

N = X_in.size(0)
nTrain = int(N*0.9)

sudoku_train = TensorDataset(X[:nTrain], is_input[:nTrain], Y[:nTrain])
sudoku_test =  TensorDataset(X[nTrain:], is_input[nTrain:], Y[nTrain:])
perm_train = TensorDataset(X[:nTrain,perm], is_input[:nTrain,perm], Y[:nTrain,perm])
perm_test =  TensorDataset(X[nTrain:,perm], is_input[nTrain:,perm], Y[nTrain:,perm])