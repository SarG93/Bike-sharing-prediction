import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from bike_count_prediction.args import load_args
from bike_count_prediction.data.data import BikeSharingDataset

if __name__ == '__main__':
    # Train model
    args = load_args()

    dataset = BikeSharingDataset(args)
    X = np.array([val[0] for val in list(dataset.data_dict.values())])
    y = np.array(np.array([sum(val[1]) for val in list(dataset.data_dict.values())]))

    RFreg = RandomForestRegressor().fit(X, y)

    # evaluate model
    args = load_args()
    args.batch_size = 1
    args.train = False

    dataset = BikeSharingDataset(args)
    X = np.array([val[0] for val in list(dataset.data_dict.values())])
    y = np.array(np.array([sum(val[1]) for val in list(dataset.data_dict.values())]))

    prediction = np.around(RFreg.predict(X))

    print(f'Mean Expected Error: {np.mean(np.absolute(prediction - y))}')
    plt.scatter(prediction, y, alpha=0.4)
    plt.plot(range(1000), range(1000), 'r')
    plt.xlabel('True count')
    plt.ylabel('Predicted count')
    plt.title('Random Forest Regressor')
    plt.savefig('scatter_plot_RF.png')
    plt.show()
