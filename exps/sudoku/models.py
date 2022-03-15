import torch
import torch.nn as nn
import torch.nn.functional as F

import satnet
import infogan.models.mnist_model as mnist_infogan
import infogan.models.fashionmnist_model as fashion_infogan



class SudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m, leak_labels, softmax, pass_mask=False):
        super(SudokuSolver, self).__init__()
        self.n = boardSz
        self.pass_mask = pass_mask

        self.sat = satnet.SATNet(self.n*2 if self.pass_mask else self.n, m, aux, leak_labels=leak_labels)
        self.softmax = softmax

    def forward(self, y_in, mask):

        in_ = torch.cat((y_in, mask.float()), dim=1) if self.pass_mask else y_in
        if self.pass_mask:
            mask = torch.cat((mask, torch.ones_like(mask)), dim=1)

        out = self.sat(in_, mask)

        if self.pass_mask:
            out = out[:, :self.n]

        if self.softmax:
            out = torch.softmax(out.contiguous().view(-1, 9), dim=1).view(-1, in_.shape[1])

        return out, y_in

    def get_pieces(self):
        return {'satnet': self.state_dict()}

    def load_from_pieces_if_present(self, load):
        if 'satnet' in load:
            print(f'Found satnet model in file. Loading...')
            self.load_state_dict(load['satnet'])

    

class Proofreader(nn.Module):
    def __init__(self):
        super().__init__()

        self.proofreader = nn.Linear(9**3, 9**3)
        torch.nn.init.eye_(self.proofreader.weight)

        OFFSET = 10
        self.proofreader.weight.data *= OFFSET
        torch.nn.init.constant_(self.proofreader.bias, -OFFSET/2)

        NOISE_SCALE = 1e-4
        self.proofreader.weight.data += torch.randn_like(self.proofreader.weight)*NOISE_SCALE
        self.proofreader.bias.data += torch.randn_like(self.proofreader.bias)*NOISE_SCALE

    def forward(self, y_in):
        return F.softmax(self.proofreader(y_in).view(-1, 9), dim=1).view(-1, 9**3)

    def get_pieces(self):
        return {'proofreader': self.state_dict()}

    def load_from_pieces_if_present(self, load):
        if 'proofreader' in load:
            print(f'Found proofreader model in file. Loading...')
            self.load_state_dict(load['proofreader'])

class ProofreadSudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m, leak_labels, softmax):
        super().__init__()
        self.sudoku_solver = SudokuSolver(boardSz, aux, m, leak_labels, softmax=softmax)
        self.proofreader = Proofreader()

    def forward(self, y_in, mask):
        out = self.sudoku_solver(self.proofreader(y_in), mask)
        return out

    def get_pieces(self):
        return {**self.sudoku_solver.get_pieces(), **self.proofreader.get_pieces()}

    def load_from_pieces_if_present(self, load):
        self.sudoku_solver.load_from_pieces_if_present(load)
        self.proofreader.load_from_pieces_if_present(load)


class DigitConv(nn.Module):
    '''
    Convolutional neural network for MNIST digit recognition. From:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    '''
    def __init__(self):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)[:,:9].contiguous()

    def get_pieces(self):
        return {'lenet': self.state_dict()}

    def load_from_pieces_if_present(self, load):
        if 'lenet' in load:
            print(f'Found lenet digit classifier in file. Loading...')
            self.load_state_dict(load['lenet'])

class MNISTSudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m, leak_labels, softmax, proofread=False):
        super(MNISTSudokuSolver, self).__init__()
        self.digit_convnet = DigitConv()

        solver = SudokuSolver if not proofread else ProofreadSudokuSolver
        self.sudoku_solver = solver(boardSz, aux, m, leak_labels, softmax=softmax)
        self.boardSz = boardSz
        self.nSq = boardSz

        self.current_perm = None
    
    def forward(self, x, is_inputs):
        nBatch = x.shape[0]
        x = x.flatten(start_dim = 0, end_dim = 1)

        digit_guess = self.digit_convnet(x)
        puzzles = digit_guess.view(nBatch, self.nSq)

        solution = self.sudoku_solver(puzzles, is_inputs)
        return solution

    def get_pieces(self):
        return {**self.sudoku_solver.get_pieces(), **self.digit_convnet.get_pieces(), 'current_perm': self.current_perm}

    def load_from_pieces_if_present(self, load):
        self.sudoku_solver.load_from_pieces_if_present(load)
        self.digit_convnet.load_from_pieces_if_present(load)

        if 'current_perm' in load:
            print(f'Found permutation matrix in file. Loading...')
            print(load['current_perm'])
            self.current_perm = load['current_perm']

        

class InfoGanSudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m, leak_labels, fashion=False):
        super(InfoGanSudokuSolver, self).__init__()

        infogan_module = fashion_infogan if fashion else mnist_infogan
        self.discriminator = infogan_module.Discriminator()
        self.qnet = infogan_module.QHead()
        self.sudoku_solver = SudokuSolver(boardSz, aux, m, leak_labels, softmax=True, pass_mask=False)
        self.boardSz = boardSz
        self.nSq = boardSz

        self.proofreader = None

        self.current_perm = None

    def initialize_proofreader(self):
        self.proofreader = Proofreader()

    def load_infogan_save(self, path):
        
        state_dict = torch.load(path)

        self.discriminator.load_state_dict(state_dict['discriminator'])
        self.qnet.load_state_dict(state_dict['netQ'])
    
    def forward(self, x, is_inputs):
        nBatch = x.shape[0]
        x = x.view(-1, 1, 28, 28)
        
        c1, c2, c3 = self.qnet(self.discriminator(x))

        exp_c1 = torch.exp(c1)
        probs = exp_c1/torch.sum(exp_c1, dim=1, keepdim=True)

        probs = probs.view(nBatch, self.nSq)

        if self.proofreader is not None:
            probs = self.proofreader(probs)

        solution = self.sudoku_solver(probs, is_inputs)
        return solution


    def get_pieces(self):
        return {**self.sudoku_solver.get_pieces(), 'discriminator': self.discriminator.state_dict(), 'netQ': self.qnet.state_dict(), 'current_perm': self.current_perm}

    def load_from_pieces_if_present(self, load):
        self.sudoku_solver.load_from_pieces_if_present(load)

        if 'discriminator' in load:
            print(f'Found discrimator model in file. Loading...')
            self.discriminator.load_state_dict(load['discriminator'])
        
        if 'netQ' in load:
            print(f'Found netQ model in file. Loading...')
            self.qnet.load_state_dict(load['netQ'])

        if 'current_perm' in load:
            print(f'Found permutation matrix in file. Loading...')
            print(load['current_perm'])
            self.current_perm = load['current_perm']

