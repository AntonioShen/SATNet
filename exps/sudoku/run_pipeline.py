import argparse
from pathlib import Path

from run import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='sudoku')
    parser.add_argument('--load-model', type=str)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--leak-labels', action='store_true')
    parser.add_argument('--mode', required=False)
    parser.add_argument('--num-errors', type=int, default=0)
    parser.add_argument('--solvability', choices=['solvable', 'unsolvable', 'any'], default='any')
    parser.add_argument('--infogan-labels-dir', type=str)
    parser.add_argument('--num-cats', type=int, default=9)

    args = parser.parse_args()
    infogan_path = args.load_model

    # args.load_model = 'exps/sudoku/infogan/num_cats_{}/model_epoch_MNIST_50'.format(str(args.num_cats))
    # print(f'Extracting Perm...')
    # args.mode = 'train-satnet-visual-infogan'
    # infogan_satnet_model = run(args, num_experiment_repetitions=1, num_epochs=2)[0]
    #
    # args.load_model = 'num_cats_{}/logs/sudoku.train-satnet-visual-infogan.boardSz3-aux300-m600-lr0.002-bsz300-exp0' \
    #                   '/it2.pth'.format(
    #     str(args.num_cats))
    # print(f'Generating Dataset...')
    # args.mode = 'satnet-visual-infogan-generate-dataset'
    # run(args, num_experiment_repetitions=1)
    #
    # args.load_model = 'num_cats_{}/logs/sudoku.train-satnet-visual-infogan.boardSz3-aux300-m600-lr0.002-bsz300-exp0' \
    #                   '/it2.pth'.format(
    #     str(args.num_cats))
    # print(f'Distilling...')
    # args.mode = 'train-backbone-lenet-supervised'
    # args.infogan_labels_dir = 'num_cats_{}'.format(str(args.num_cats))
    # # args.load_model = Path(infogan_satnet_model)/'it2.pth'
    # distilled_model = run(args, num_experiment_repetitions=1, num_epochs=1)[0]

    args.load_model = 'num_cats_{}/logs/sudoku.train-backbone-lenet-supervised.boardSz3-aux300-m600-lr0.002-bsz40' \
                      '-exp0/it1.pth'.format(
        str(args.num_cats))
    print(f'Training SATNet E2E...')
    args.mode = 'visual'
    # args.load_model = Path(distilled_model)/'it1.pth'
    args.infogan_labels_dir = None
    distilled_model = run(args, num_experiment_repetitions=1, num_epochs=100)[0]
