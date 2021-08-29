from bike_count_prediction.args import load_args
from bike_count_prediction.data.data import get_dataloader
from bike_count_prediction.model import MLPBikePrediction
from bike_count_prediction.train import train
from split_dataset import DatasetSplit

if __name__ == '__main__':
    # Load arguments
    args = load_args()
    # Split dataset
    if args.split_dataset:
        split = DatasetSplit()
        split.train_test_split()
    # Instantiate model
    model = MLPBikePrediction()
    # create dataloader
    dataloader = get_dataloader(args)
    # Train and save model
    train(args, model, dataloader)
