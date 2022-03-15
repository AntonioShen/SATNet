import torch
import torch.nn.functional as F


def to_oh(x):
    x_max = torch.argmax(x, dim=1).view(x.shape[0], 1)
    out = torch.zeros_like(x).scatter_(1, x_max, 1)

    return out


def permutation_invariant_loss(x, y):
    '''
    x is pred, y is target
    '''
    x = x.view(-1, 9)
    y = y.view(-1, 9)

    p = torch.zeros((9, 9))

    if x.is_cuda: p = p.cuda()

    for n in range(9):
        for m in range(9):
            p[n, m] = torch.exp(-F.binary_cross_entropy(x[:, n], y[:, m]))

    p_res = torch.sqrt(torch.sum(p ** 2, dim=1))
    p_1 = (x.shape[0] / 9 - torch.mean(p_res)) / (x.shape[0] / 9)

    p_oh = to_oh(p)  # just converts p to a one-hot form sot that it becomes a true permutation matrix

    return p_1, p_oh, p


def zyy_loss(x, y, num_cats):
    '''
    x is pred, y is target
    '''
    x = x.view(-1, num_cats)
    y = y.view(-1, 9)

    p = torch.zeros((num_cats, 9))

    if x.is_cuda: p = p.cuda()

    for n in range(num_cats):
        for m in range(9):
            p[n, m] = torch.exp(-F.binary_cross_entropy(x[:, n], y[:, m]))

    p_res = torch.sqrt(torch.sum(p ** 2, dim=1))
    p_1 = (x.shape[0] / num_cats - torch.mean(p_res)) / (x.shape[0] / num_cats)
    p_oh = to_oh(p)  # just converts p to a one-hot form sot that it becomes a true permutation matrix

    return p_1, p_oh, p
