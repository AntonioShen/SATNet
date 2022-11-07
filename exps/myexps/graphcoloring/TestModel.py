import getopt
import sys
import torch
import torch.optim as optim

from TrainGraph import test_model, get_dataset_cuda, FigLogger
from exps.myexps.graphcoloring.GraphSolver import GraphColoringSolver
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt


def main(argv):
    state_dict = ''
    data_path = ''
    n_test_r = 0.1

    try:
        opts, args = getopt.getopt(argv, "hf:d:n:", ["model-file=", "dataset=", "n-test="])
    except getopt.GetoptError:
        print('TrainGraph.py --model-file <path> --dataset <path> --n-test <float>')
    for opt, arg in opts:
        if opt == '-h':
            print('TrainGraph.py --model-file <path> --dataset <path> --n-test <int>')
            exit()
        elif opt in ("-f", "--model-file"):
            state_dict = arg
        elif opt in ("-d", "--dataset"):
            data_path = arg
        elif opt in ("-n", "--n-test"):
            n_test_r = float(arg)

    v_num = 10
    aux = 600
    m = 1500
    lr = 2e-3
    batchSz = 40
    nEpoch = 10

    X_in, Y_in, is_input = get_dataset_cuda(data_path)
    N = X_in.size(0)
    n_train = int(N * (1 - n_test_r))
    print(X_in[n_train:].shape, is_input[n_train:].shape, Y_in[n_train:].shape)
    graph_coloring_test = TensorDataset(X_in[n_train:], is_input[n_train:], Y_in[n_train:])

    graph_coloring_model = GraphColoringSolver(v_num, aux, m)
    graph_coloring_model.load_state_dict(torch.load(state_dict))
    graph_coloring_model = graph_coloring_model.cuda()

    optimizer = optim.Adam(graph_coloring_model.parameters(), lr=lr)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.4)
    test_logger = FigLogger(fig, axes[1], 'Testing')

    test_model(v_num, aux, m, lr, batchSz, nEpoch, graph_coloring_model, optimizer, test_logger, graph_coloring_test)


if __name__ == "__main__":
    main(sys.argv[1:])