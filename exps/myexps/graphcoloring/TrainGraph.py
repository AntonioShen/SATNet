import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from GraphSolver import GraphColoringSolver
from torch.utils.data import TensorDataset, DataLoader


class FigLogger(object):
    def __init__(self, fig, base_ax, title):
        self.colors = ['tab:red', 'tab:blue']
        self.labels = ['Loss (entropy)', 'Error']
        self.markers = ['d', '.']
        self.axes = [base_ax, base_ax.twinx()]
        base_ax.set_xlabel('Epochs')
        base_ax.set_title(title)

        for i, ax in enumerate(self.axes):
            ax.set_ylabel(self.labels[i], color=self.colors[i])
            ax.tick_params(axis='y', labelcolor=self.colors[i])

        self.reset()
        self.fig = fig

    def log(self, args):
        for i, arg in enumerate(args[-2:]):
            self.curves[i].append(arg)
            x = list(range(len(self.curves[i])))
            self.axes[i].plot(x, self.curves[i], self.colors[i], marker=self.markers[i])
            self.axes[i].set_ylim(0, 1.05)

        self.fig.canvas.draw()

    def reset(self):
        for ax in self.axes:
            for line in ax.lines:
                line.remove()
        self.curves = [[], []]


def get_dataset_cuda(file_path):
    data_tensor = torch.load(file_path)
    X = data_tensor[0]
    Y = data_tensor[1]
    is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()
    is_input = is_input.view(is_input.size(0), -1)
    is_input = is_input.cuda()
    X = X.view(X.size(0), -1)
    Y = Y.view(Y.size(0), -1)
    X = X.cuda()
    Y = Y.cuda()
    return X, Y, is_input


def run(v_num, epoch, model, optimizer, logger, dataset, batchSz, to_train=False, unperm=None):
    loss_final, err_final = 0, 0

    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))

    for i, (data, is_input_, label) in tloader:
        if to_train: optimizer.zero_grad()
        preds = model(data.contiguous(), is_input_.contiguous())
        label = label.float()
        loss = nn.functional.binary_cross_entropy(preds, label)
        if to_train:
            loss.backward()
            optimizer.step()
        err = computeErr(preds.data, v_num, unperm, label) / batchSz
        tloader.set_description(
            'Epoch {} {} Loss {:.4f} Err: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), err))
        loss_final += loss.item()
        err_final += err

    loss_final, err_final = loss_final / len(loader), err_final / len(loader)
    logger.log((epoch, loss_final, err_final))

    if not to_train:
        print('TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_final, err_final))

    # print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
    torch.cuda.empty_cache()


def train(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, True, unperm)


@torch.no_grad()
def test(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, False, unperm)


@torch.no_grad()
def computeErr(pred_flat, n, unperm, label_flat):
    pred = pred_flat.view(-1, n, n, n + 1)
    label = label_flat.view(-1, n, n, n + 1)
    batch_size = pred.size(0)
    correct_count = 0

    def convert_argmax_catalog(mat_prob):
        mat_catalog = torch.zeros(n, n, n + 1)
        mat_catalog = mat_catalog.cuda()
        for i in range(mat_prob.size(0)):
            for j in range(mat_prob.size(1)):
                arg_max = torch.argmax(mat_prob[i][j])
                mat_catalog[i][j][arg_max] = 1
        return mat_catalog

    for i in range(batch_size):
        p = convert_argmax_catalog(pred[i])
        l = label[i]
        if torch.equal(p, l):
            correct_count = correct_count + 1

    return float(batch_size - correct_count)

    # if unperm is not None: pred_flat[:, :] = pred_flat[:, unperm]
    #
    # nsq = n
    # pred = pred_flat.view(-1, nsq, nsq, nsq + 1)
    #
    # batchSz = pred.size(0)
    # s = (nsq - 1) * nsq // 2  # 0 + 1 + ... + n^2-1
    # I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq)
    #
    # def invalidGroups(x):
    #     valid = (x.min(1)[0] == 0)
    #     valid *= (x.max(1)[0] == nsq - 1)
    #     valid *= (x.sum(1) == s)
    #     return valid.bitwise_not()
    #
    # boardCorrect = torch.ones(batchSz).type_as(pred)
    # for j in range(nsq):
    #     # Check the jth row and column.
    #     boardCorrect[invalidGroups(I[:, j, :])] = 0
    #     boardCorrect[invalidGroups(I[:, :, j])] = 0
    #
    #     # Check the jth block.
    #     row, col = n * (j // n), n * (j % n)
    #     M = invalidGroups(I[:, row:row + n, col:col + n].contiguous().view(batchSz, -1))
    #     boardCorrect[M] = 0
    #
    #     if boardCorrect.sum() == 0:
    #         return batchSz

    # return float(batchSz - boardCorrect.sum())


v_num = 6
aux = 300
m = 600
lr = 2e-3
batchSz = 40
nEpoch = 100

X_in, Y_in, is_input = get_dataset_cuda("production/VNUM_" + str(v_num) + "_DATA_ARRAY.pt")
N = X_in.size(0)
n_train = int(N*0.9)
print(X_in.shape, Y_in.shape, is_input.shape)
graph_coloring_train = TensorDataset(X_in[:n_train], is_input[:n_train], Y_in[:n_train])
graph_coloring_test = TensorDataset(X_in[n_train:], is_input[n_train:], Y_in[n_train:])
graph_coloring_model = GraphColoringSolver(v_num, aux, m)
graph_coloring_model = graph_coloring_model.cuda()

plt.ioff()
optimizer = optim.Adam(graph_coloring_model.parameters(), lr=lr)

fig, axes = plt.subplots(1, 2, figsize=(10,4))
plt.subplots_adjust(wspace=0.4)
train_logger = FigLogger(fig, axes[0], 'Traininig')
test_logger = FigLogger(fig, axes[1], 'Testing')

test(v_num, 0, graph_coloring_model, optimizer, test_logger, graph_coloring_test, batchSz)
plt.pause(0.01)
for epoch in range(1, nEpoch + 1):
    train(v_num, epoch, graph_coloring_model, optimizer, train_logger, graph_coloring_train, batchSz)
    test(v_num, epoch, graph_coloring_model, optimizer, test_logger, graph_coloring_test, batchSz)

