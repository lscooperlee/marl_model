import argparse

from .executes import run


def main():
    parser = argparse.ArgumentParser(description='Run Recursive MARL model')
    parser.add_argument('--model_path', default='/tmp/model_path', dest='model_path', help='path to saved training data')
    parser.add_argument('--action', dest='action', choices=['evaluate', 'train', 'replay'], default='replay', help='action')

    args = parser.parse_args()
    if args.action == 'train':
        run.train(model_path=args.model_path)
    elif args.action == 'evaluate':
        run.evaluate(model_path=args.model_path)
    elif args.action == 'replay':
        run.replay(model_path=args.model_path)


if __name__ == '__main__':
    main()