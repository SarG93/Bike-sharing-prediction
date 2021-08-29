import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--learning_rate', default=0.003, type=float)

parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--device', default='cpu', type=str)

parser.add_argument('--train', default=True, type=bool)

parser.add_argument('--datapath', default='./bike_count_prediction/data/bike_sharing_dataset', type=str)
parser.add_argument('--modelpath', default='./models', type=str)
parser.add_argument('--split_dataset', default=True, type=bool)


def load_args():
    return parser.parse_args()
