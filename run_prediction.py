import os

import torch

from bike_count_prediction.args import load_args
from bike_count_prediction.data.data import get_dataloader
from bike_count_prediction.model import MLPBikePrediction
from bike_count_prediction.train import evaluate
from scripts.split_dataset import DatasetSplit

if __name__ == '__main__':
    # Load args
    args = load_args()
    args.batch_size = 1
    args.train = False
    if args.split_dataset:
        split = DatasetSplit()
        split.train_test_split()
    # Load model
    model = MLPBikePrediction()
    model.load_state_dict(torch.load(os.path.join('./models', 'bike_count_mlp.dat'), map_location=args.device))
    # create dataloader from test data
    dataloader = get_dataloader(args, shuffle=False)
    # evaluate model performance on test data 
    evaluate(args, model, dataloader)
