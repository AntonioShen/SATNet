from GraphSolver import GraphColoringSolver
import getopt
import sys
import torch
from torch.utils.data import TensorDataset
import torch.optim as optim
from matplotlib import pyplot as plt
from TrainGraph import FigLogger, train, test


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


def main(argv):
    saved_dir = ''
    epoch_start = 0
    try:
        opts, args = getopt.getopt(argv, "hl:e:", ['help', 'load=', 'epoch='])
    except getopt.GetoptError:
        print('Usage: python ContinueTrain.py -l <torch model>')
        exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('Usage: python ContinueTrain.py -l <torch model> -e <start epoch>')
            exit()
        elif opt in ('-l', '--load'):
            saved_dir = arg
        elif opt in ('-e', '--epoch'):
            epoch_start = arg

    v_num = 8
    aux = 600
    m = 300
    lr = 2e-3
    batchSz = 40
    nEpoch = 100

    X_in, Y_in, is_input = get_dataset_cuda("production/VNUM_" + str(v_num) + "_DATA_ARRAY.pt")
    N = X_in.size(0)
    n_train = int(N * 0.9)
    print(X_in.shape, Y_in.shape, is_input.shape)
    graph_coloring_train = TensorDataset(X_in[:n_train], is_input[:n_train], Y_in[:n_train])
    graph_coloring_test = TensorDataset(X_in[n_train:], is_input[n_train:], Y_in[n_train:])
    graph_coloring_model = GraphColoringSolver(v_num, aux, m)
    graph_coloring_model.load_state_dict(torch.load(saved_dir))
    graph_coloring_model = graph_coloring_model.cuda()
    plt.ioff()
    optimizer = optim.Adam(graph_coloring_model.parameters(), lr=lr)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.4)
    train_logger = FigLogger(fig, axes[0], 'Traininig')
    test_logger = FigLogger(fig, axes[1], 'Testing')
    for epoch in range(int(epoch_start), nEpoch + 1):
        train(v_num, epoch, graph_coloring_model, optimizer, train_logger, graph_coloring_train, batchSz)
        test(v_num, epoch, graph_coloring_model, optimizer, test_logger, graph_coloring_test, batchSz)
        if (epoch % 20) == 0:
            torch.save(graph_coloring_model.state_dict(), './production/weights_VNUM_' + str(v_num) + '_AUX_'
                       + str(aux) + '_M_' + str(m) + '_LR_' + str(lr) + '_BATCSZ_' + str(batchSz) + '_EP_' + str(
                epoch) + '.pt')


if __name__ == "__main__":
    main(sys.argv[1:])
