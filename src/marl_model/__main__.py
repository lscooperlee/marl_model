import os
import argparse
from datetime import datetime

from .executes import run


def main():
    parser = argparse.ArgumentParser(description='Run Recursive MARL model')
    subparsers = parser.add_subparsers(help='sub-commands help')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--model_path', default='', help='path to saved training data')
    parser_train.add_argument('--model', choices=['dqn', 'cdqn'], default='dqn', help='model')
    parser_train.add_argument('--shape', choices=['1x9', '3x3', '5x5', '9x9', '10x10'], default='1x9', help='shape')
    parser_train.add_argument('--env', choices=['mtns', 'mtr', 'mts'], default='mtr', help='env')
    parser_train.add_argument('--map',
                              choices=['simple', 'prob', 'simple_update', 'prob_update', 'none'],
                              default='prob',
                              help='map')
    parser_train.add_argument('--resume', choices=['all', 'model'], help='path to saved training data', default=None)
    parser_train.add_argument('--fix_task', help='path to saved training data', default=True, action='store_true')

    def _train(args):
        folder_name = f'{args.model}_{args.env}_{args.shape}_{args.map}_{args.fix_task}_{datetime.now():%Y%m%d_%H%M}'
        # model_path = args.resume if args.resume else os.path.join(args.model_path, folder_name)
        model_path = args.model_path if args.resume else os.path.join(args.model_path, folder_name)
        env_shape = tuple([int(x) for x in args.shape.split('x')])
        run.train(model_path=model_path,
                  qmodel=args.model,
                  env_shape=env_shape,
                  env_map=args.map,
                  env_name=args.env,
                  resume_mode=args.resume,
                  fix_task=args.fix_task)

    parser_train.set_defaults(func=_train)

    parser_eval = subparsers.add_parser('evaluate')
    parser_eval.add_argument('--model_path', default='', help='path to saved training data')

    def _eval(args):
        run.evaluate(args.model_path)

    parser_eval.set_defaults(func=_eval)

    parser_replay = subparsers.add_parser('replay')
    parser_replay.add_argument('--model_path', default='', help='path to saved training data')
    parser_replay.add_argument('--truck', default=None, help='truck_loc')
    parser_replay.add_argument('--map',
                               choices=['simple', 'prob', 'simple_update', 'prob_update', 'none'],
                               default=None,
                               help='map')

    def _replay(args):
        truck_loc = [int(x) for x in args.truck.split(',')] if args.truck else args.truck
        run.replay(args.model_path, truck_loc, args.map)

    parser_replay.set_defaults(func=_replay)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()