import os

import pandas as pd

DATAPATH = './bike_count_prediction/data/bike_sharing_dataset'


class DatasetSplit:

    def __init__(self, train_ratio: float = 0.7):
        self.df_hour = pd.read_csv(os.path.join(DATAPATH, 'hour.csv'))
        self.df_day = pd.read_csv(os.path.join(DATAPATH, 'day.csv'))

        date_set = self.df_day['dteday'].unique().tolist()

        self.test_dates = date_set[int(train_ratio * len(date_set)):]
        self.train_dates = date_set[:int(train_ratio * len(date_set))]

    def __filter_and_save(self, dates: list, folder_name: str):
        save_path = os.path.join(DATAPATH, folder_name)
        os.makedirs(save_path, exist_ok=True)

        df_hour = self.df_hour[self.df_hour['dteday'].isin(dates)]
        df_hour.to_csv(os.path.join(save_path, 'hour.csv'))

        df_day = self.df_day[self.df_day['dteday'].isin(dates)]
        df_day.to_csv(os.path.join(save_path, 'day.csv'))

    def train_test_split(self):
        self.__filter_and_save(self.test_dates, 'test')
        self.__filter_and_save(self.train_dates, 'train')


if __name__ == '__main__':
    split = DatasetSplit()
    split.train_test_split()
