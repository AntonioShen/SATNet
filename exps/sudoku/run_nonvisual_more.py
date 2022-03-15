import argparse

import main_nonvisual_more as main


def run(args, num_experiment_repetitions=1, num_epochs=100):
    batch_size = 40
    if args.mode == 'train-satnet-visual-infogan': batch_size = 300

    lr = 2e-3

    results = []
    for i in range(num_experiment_repetitions):
        print(f'STARTING EXPERIMENT {i}')
        result = main.main(
            data_dir=args.data_dir,
            load_model=args.load_model,
            no_cuda=args.no_cuda,
            to_train=not args.no_train,
            mode=args.mode,
            leak_labels=args.leak_labels,
            lr=lr,
            nEpoch=num_epochs,
            batchSz=batch_size,
            solvability=args.solvability,
            num_injected_input_cell_errors=args.num_errors,
            infogan_labels_dir=args.infogan_labels_dir,
            experiment_num=i,
            num_cats=args.num_cats
        )
        results.append(result)

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='sudoku')
    parser.add_argument('--load-model', type=str)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--leak-labels', action='store_true')
    parser.add_argument('--mode', choices=main.MODES, required=True)
    parser.add_argument('--num-errors', type=int, default=0)
    parser.add_argument('--solvability', choices=['solvable', 'unsolvable', 'any'], default='any')
    parser.add_argument('--infogan-labels-dir', type=str)
    parser.add_argument('--num-experiments', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-cats', type=int, default=18)

    args = parser.parse_args()

    run(args, args.num_experiments, args.num_epochs)