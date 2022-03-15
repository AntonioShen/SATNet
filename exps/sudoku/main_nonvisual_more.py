#!/usr/bin/env python3
#
# Partly derived from:
#   https://github.com/locuslab/optnet/blob/master/sudoku/train.py

import argparse
import signal
import sys
import os
import shutil
import csv
import random

import numpy as np
import numpy.random as npr
# import setproctitle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from visualize import visualize

# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=1000)

import prep_dataset
import models_unkcats
from loss import permutation_invariant_loss, to_oh, zyy_loss

MODES = [
    'nonvisual',
    'visual',
    'train-proofreader-nonvisual',
    'train-satnet-visual-infogan',
    'satnet-visual-infogan-generate-dataset',
    'train-backbone-lenet-supervised',
    'train-proofreader-lenet',
]


class CSVLogger(object):
    def __init__(self, fname):
        self.f = open(fname, 'w')
        self.logger = csv.writer(self.f)

    def log(self, fields):
        self.logger.writerow(fields)
        self.f.flush()


def print_header(msg):
    print('===>', msg)


def setup_state(cuda):
    # For debugging: fix the random seed
    # npr.seed(1)
    # torch.manual_seed(7)
    # random.seed(7)

    if cuda:
        print('Using', torch.cuda.get_device_name(0))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()


def freeze_model(model):
    print(f'freezing model {model.__class__.__name__}.')
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    print(f'unfreezing model {model.__class__.__name__}.')

    for param in model.parameters():
        param.requires_grad = True


def main(
        data_dir='sudoku',
        boardSz=3,
        batchSz=40,
        testBatchSz=40,
        aux=300,
        m=600,
        nEpoch=100,
        lr=2e-3,
        load_model=None,
        load_infogan=None,
        no_cuda=False,
        mode=None,
        to_train=True,
        leak_labels=False,
        num_injected_input_cell_errors=0,
        solvability='any',
        infogan_labels_dir=None,
        experiment_num=0,
        num_cats=18,
):
    assert mode in MODES
    cuda = not no_cuda and torch.cuda.is_available()
    setup_state(cuda)

    save = 'sudoku.{}.boardSz{}-aux{}-m{}-lr{}-bsz{}-exp{}'.format(mode,
                                                                   boardSz, aux, m, lr, batchSz, experiment_num)
    save = os.path.join('num_cats_{}/logs'.format(str(num_cats)), save)
    if os.path.isdir(save): shutil.rmtree(save)
    os.makedirs(save)

    print_header(f'Run saving to {save}')

    print_header('Loading data')

    perm = None
    # zyy
    # perm = np.zeros((boardSz ** 4, boardSz ** 2))
    # for i in range(0, perm.shape[0]):
    #     perm[i, :] = i * 9
    #     perm[i, :] += np.array([1, 2, 3, 4, 5, 6, 7, 8, 0])
    #     perm[i, :] += np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    #
    # perm = torch.tensor(perm.reshape(-1), dtype=int)

    if mode in (
            'nonvisual',
            'train-proofreader-nonvisual',
    ):
        train_set, test_set, unperm = prep_dataset.get_sudoku_nonvisual(
            data_dir, cuda, perm, num_injected_input_cell_errors, solvability
        )

    print_header('Building model')

    if mode in ('nonvisual',):
        model = models_unkcats.SudokuSolver(boardSz**4 * num_cats, aux, m, leak_labels, softmax=True)


        if load_model:
            model.load_from_pieces_if_present(torch.load(load_model))

    if cuda: model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_logger = CSVLogger(os.path.join(save, 'train.csv'))
    test_logger = CSVLogger(os.path.join(save, 'test.csv'))
    fields = ['epoch', 'loss', 'total_board_error', 'per_cell_error', 'digit_classification_error']
    train_logger.log(fields)
    test_logger.log(fields)

    img_list_train = []
    img_list_test = []

    def render_img_list(img_list, file_name):
        # Animation showing the improvements of the generator.
        fig = plt.figure(figsize=(10, 10))
        plt.axis("off")
        ims = [[plt.imshow(i, animated=True)] for i in img_list]
        anim = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000, blit=True)
        anim.save(file_name, dpi=80, writer='imagemagick')

    def signal_handler(sig, frame):

        render_img_list(img_list_train, 'perm_train.gif')
        render_img_list(img_list_test, 'perm_test.gif')

        sys.exit(0)



    # run_extract(boardSz**4 * num_cats, 0, model, optimizer, test_logger, train_set, testBatchSz, unperm, mode)
    label_set = test(boardSz, 0, model, optimizer, test_logger, test_set, testBatchSz, unperm, mode, img_list_test,
                     leak_labels, num_cats)

    if to_train:
        for epoch in range(1, nEpoch + 1):
            train(boardSz, epoch, model, optimizer, train_logger, train_set, batchSz, unperm, mode, img_list_train,
                  leak_labels, num_cats)
            test(boardSz, epoch, model, optimizer, test_logger, test_set, testBatchSz, unperm, mode, img_list_test,
                 leak_labels, num_cats)
            torch.save(model.get_pieces(), os.path.join(save, 'it' + str(epoch) + '.pth'))


    return save


def run(boardSz, epoch, model, optimizer, logger, dataset, batchSz, to_train, unperm, mode, img_list, leak_labels,
        num_cats):
    loss_final, board_err_final, err_solvable_final, err_unsolvable_final, solvable_total, unsolvable_total, cell_err_final, visual_err_final, num_inputs_final = 0., 0., 0., 0., 0., 0., 0., 0., 0.

    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))

    if mode in ('satnet-visual-infogan-generate-dataset',):
        label_set = []

    avg_perm = torch.zeros(len(tloader), num_cats, 9)
    for i, content in tloader:
        data, is_input, label = content[:3]
        # zyy

        # zyy
        # is_input_original = torch.clone(is_input)
        data = data.cpu().detach().numpy()
        data = data.reshape((-1, boardSz ** 2))
        data_new = np.zeros((data.shape[0], num_cats))
        for j in range(data.shape[0]):
            # if random.random() > 0.5:
            #     data_new[j, 9:18] = data[j, :]
            # else:
            #     data_new[j, :9] = data[j, :]
            #     pass
            data_new[j, 9:18] = 0.5 * data[j, :]
            data_new[j, :9] = 0.5 * data[j, :]
            pass
        data = torch.from_numpy(data_new.reshape((batchSz, boardSz ** 4 * num_cats))).cuda().type(torch.float32)

        is_input = torch.reshape(is_input, (batchSz, boardSz ** 2, boardSz ** 2, boardSz ** 2))
        is_input_add, _ = torch.topk(is_input, num_cats - 9)
        is_input = torch.cat((is_input, is_input_add), 3)
        is_input = torch.reshape(is_input, (batchSz, boardSz ** 2 * boardSz ** 2 * num_cats))

        # visualize(data[0].cpu().detach().numpy(), is_input[0].cpu().detach().numpy(),
        #           label[0][unperm].cpu().detach().numpy())
        # exit(0)

        # FIXME vvvv
        solvable = content[3] if len(content) > 3 and mode not in ('train-backbone-lenet-supervised',) else None

        is_input_mono = is_input.view(-1, num_cats).float().mean(dim=1).type(torch.bool)
        # is_input_mono = is_input_original.view(-1, 9).float().mean(dim=1).type(torch.bool)

        if to_train: optimizer.zero_grad()

        preds, reconstructed = model(data.contiguous(), is_input.contiguous())
        preds = preds.contiguous()

        # zyy
        # print(data.contiguous().shape)
        # print(is_input.contiguous().shape)
        # print(preds.shape)
        # print(reconstructed.shape)
        # break

        if mode in ('nonvisual'):
            # perm_oh = None
            if hasattr(model, 'current_perm') and model.current_perm is not None:
                perm_oh = model.current_perm.cuda()

            # zyy
            perm_oh = torch.tensor([
                [1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.]
            ]).cuda()

            preds = preds * (1. - is_input.float())
            preds = preds + reconstructed * is_input.float()

            if perm_oh is not None:
                preds = torch.matmul(preds.view(-1, num_cats), perm_oh).contiguous().view(-1, 9 ** 3)

            if leak_labels:
                preds_loss = preds
                label_loss = label
            else:
                preds_loss = preds.view(-1, 9)[~is_input_mono, :]
                label_loss = label.view(-1, 9)[~is_input_mono, :]

            # print(preds_loss)
            # print(label_loss)
            # exit(0)
            loss = F.binary_cross_entropy(preds_loss, label_loss)
            # exit(0)

            preds = to_oh(preds.view(-1, 9)).view(-1, 729)

        if to_train:
            # perm_oh[9:, :] = 0
            loss.backward()
            optimizer.step()

        board_err = computeErr(preds.data, boardSz, unperm)

        # Evaluate per-cell accuracy
        preds = preds.view(-1, 9)

        num_cells = label.shape[1] / 9
        cell_err = torch.any(preds != label.view(-1, 9), dim=1)
        visual_err = cell_err[is_input_mono].sum().float()
        num_inputs = is_input_mono.int().sum()

        cell_err = cell_err.sum().float()

        if solvable is not None:
            err_solvable_final += computeErr(preds.view(-1, 9 ** 3)[solvable], boardSz,
                                             unperm) if solvable.sum() > 0 else 0
            err_unsolvable_final += computeErr(preds.view(-1, 9 ** 3)[~solvable], boardSz, unperm) if (
                                                                                                          ~solvable).sum() > 0 else 0
            solvable_total += float(solvable.sum())
            unsolvable_total += float((~solvable).sum())

        tloader.set_description(
            'Epoch {} {} Loss {:.4f} Total Board Error: {:.4f} Per-Cell Error: {:.4f} Visual Error {:.4}'.format(epoch,
                                                                                                                 (
                                                                                                                     'Train' if to_train else 'Test '),
                                                                                                                 loss.item(),
                                                                                                                 board_err / batchSz,
                                                                                                                 cell_err / batchSz / num_cells,
                                                                                                                 visual_err / num_inputs))
        loss_final += loss.item() * batchSz
        board_err_final += board_err
        cell_err_final += cell_err
        visual_err_final += visual_err
        num_inputs_final += num_inputs

    loss_final, board_err_final, cell_err_final = loss_final / len(loader.dataset), board_err_final / len(
        loader.dataset), cell_err_final / len(loader.dataset) / num_cells
    logger.log((epoch, loss_final, board_err_final, float(cell_err_final.detach()),
                float(visual_err_final.detach() / num_inputs_final)))

    solvability_string = (
        f'Solvable: {err_solvable_final:.0f}/{solvable_total:.0f} ({100. * err_solvable_final / solvable_total if solvable_total > 0 else 0:.2f}%), Unsolvable: {err_unsolvable_final:.0f}/{unsolvable_total:.0f} ({100. * err_unsolvable_final / unsolvable_total if unsolvable_total > 0 else 0:.2f}%)'
        if err_solvable_final + err_unsolvable_final == board_err_final * len(loader.dataset) else None
    )

    title = '--------| TRAINING' if to_train else 'TESTING'
    print(
        f'{title} SET RESULTS: Average loss: {loss_final:.4f} Total Board Error: {board_err_final:.4f}, Per-Cell Error: {cell_err_final:.4f}, Visual Error: {visual_err_final / num_inputs_final:.4f}. {solvability_string}')

    # print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
    torch.cuda.empty_cache()

    # Avg perm
    avg_perm = avg_perm.mean(dim=0)
    plt.imsave(
        'num_cats_{}/avg_perm_train.png'.format(str(num_cats)) if to_train else 'num_cats_{}/avg_perm_test.png'.format(
            str(num_cats)), avg_perm.numpy())
    img_list.append(avg_perm.numpy())

    if to_train and mode in ('train-satnet-visual-infogan',):
        model.current_perm = to_oh(avg_perm).clone()

    if mode in ('satnet-visual-infogan-generate-dataset',):
        label_set = torch.cat(label_set)
        return label_set


def train(args, epoch, model, optimizer, logger, dataset, batchSz, unperm, mode, img_list, leak_labels, num_cats):
    return run(args, epoch, model, optimizer, logger, dataset, batchSz, True, unperm, mode, img_list, leak_labels,
               num_cats)


@torch.no_grad()
def test(args, epoch, model, optimizer, logger, dataset, batchSz, unperm, mode, img_list, leak_labels, num_cats):
    return run(args, epoch, model, optimizer, logger, dataset, batchSz, False, unperm, mode, img_list, leak_labels,
               num_cats)


@torch.no_grad()
def computeErr(pred_flat, n, unperm):
    if unperm is not None: pred_flat[:, :] = pred_flat[:, unperm]

    nsq = n ** 2
    pred = pred_flat.view(-1, nsq, nsq, nsq)

    batchSz = pred.size(0)
    s = (nsq - 1) * nsq // 2  # 0 + 1 + ... + n^2-1
    I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq)

    def invalidGroups(x):
        valid = (x.min(1)[0] == 0)
        valid *= (x.max(1)[0] == nsq - 1)
        valid *= (x.sum(1) == s)
        return ~valid

    boardCorrect = torch.ones(batchSz).type_as(pred)
    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:, j, :])] = 0
        boardCorrect[invalidGroups(I[:, :, j])] = 0

        # Check the jth block.
        row, col = n * (j // n), n * (j % n)
        M = invalidGroups(I[:, row:row + n, col:col + n].contiguous().view(batchSz, -1))
        boardCorrect[M] = 0

        if boardCorrect.sum() == 0:
            return batchSz

    return float(batchSz - boardCorrect.sum())
