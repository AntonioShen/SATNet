from TrainGraph import GraphColoringSolver, v_num, aux, m, nEpoch, train, test, optimizer, train_logger, test_logger, \
    graph_coloring_train, graph_coloring_test, batchSz, lr
import getopt
import sys
import torch


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
    graph_coloring_model = GraphColoringSolver(v_num, aux, m)
    graph_coloring_model.load_state_dict(torch.load(saved_dir))
    graph_coloring_model = graph_coloring_model.cuda()
    for epoch in range(epoch_start, nEpoch + 1):
        train(v_num, epoch, graph_coloring_model, optimizer, train_logger, graph_coloring_train, batchSz)
        test(v_num, epoch, graph_coloring_model, optimizer, test_logger, graph_coloring_test, batchSz)
        if (epoch % 20) == 0:
            torch.save(graph_coloring_model.state_dict(), './production/weights_VNUM_' + str(v_num) + '_AUX_'
                       + str(aux) + '_M_' + str(m) + '_LR_' + str(lr) + '_BATCSZ_' + str(batchSz) + '_EP_' + str(
                epoch) + '.pt')


if __name__ == "__main__":
    main(sys.argv[1:])
