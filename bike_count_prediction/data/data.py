import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class BikeSharingDataset(Dataset):

    def __init__(self, args):
        if args.train:
            data_path = os.path.join(args.datapath, 'train')
        else:
            data_path = os.path.join(args.datapath, 'test')
        
        self.df_hour = pd.read_csv(os.path.join(data_path, 'hour.csv'))

        self.data_dict = self.__process_data(self.df_hour)


    def __len__(self):
        return len(list(self.data_dict.keys()))


    def __getitem__(self, idx: int):
        inputs, labels = self.data_dict[idx]
        return {
            'input': np.array(inputs),
            'labels': np.array(list(labels))
        }


    def __process_data(self, df):

        def standardization_factor(lst: list):
            return np.mean(lst), np.std(lst)

        self.temp_norm = standardization_factor(self.df_hour['temp'].values)
        self.atemp_norm = standardization_factor(self.df_hour['atemp'].values)
        self.hum_norm = standardization_factor(self.df_hour['hum'].values)
        self.windspeed_norm = standardization_factor(self.df_hour['windspeed'].values)

        df['input_feature'] = df.apply(lambda x: self.extract_features(x), axis=1)
        labels = list(zip(df.casual.tolist(), df.registered.tolist()))
        processed_data = list(zip(df.input_feature.tolist(), labels))
        data_dict = dict(zip(list(range(len(processed_data))), processed_data))
        
        return data_dict


    def extract_features(self, data):

        def standardize(val, norm):
            mu, sig = norm
            return (val-mu)/sig

        def one_hot_encoding(val: int, dim:int):
            if val+1>dim:
                raise ValueError('one hot encoding not possible. value greater than the number of dimensions')
            else:
                vec = [0]*dim
                vec[val] = 1
                return vec

        feature = one_hot_encoding(data['season']-1, 4)
        feature += one_hot_encoding(data['yr'], 2)
        feature += one_hot_encoding(data['mnth']-1, 12)
        feature += one_hot_encoding(data['hr'], 24)
        feature += one_hot_encoding(data['holiday'], 2)
        feature += one_hot_encoding(data['weekday'], 7)
        feature += one_hot_encoding(data['workingday'], 2)
        feature += one_hot_encoding(data['weathersit']-1, 4)

        temp = standardize(data['temp'], self.temp_norm)
        atemp = standardize(data['atemp'], self.atemp_norm)
        hum = standardize(data['hum'], self.hum_norm)
        # windspeed = standardize(data['windspeed'], self.windspeed_norm)
        feature += [temp, atemp, hum]

        return feature


def get_dataloader(args, shuffle=True):

    dataset = BikeSharingDataset(args)
    dataloader =  DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=shuffle
        )

    return dataloader