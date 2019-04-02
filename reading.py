from __future__ import print_function, division
import torch
import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SkeletonDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        with open(csv_file, 'rb') as csvfile:
            self.landmarks_frame = list(csv.reader(csvfile))
        self.landmarks_frame = np.delete(self.landmarks_frame, 0, 0)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        landmarks = self.landmarks_frame[idx, :36].astype(float)
        target = self.landmarks_frame[idx, -1].astype(int)
        sample = [torch.Tensor(landmarks), target]

        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    # csv = pd.read_csv('Fall20_Cam4.avi_keys.csv')
    # print(type(csv.iloc[0, :36].values))

    cuda = torch.cuda.is_available()
    batch_size = 100
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    transformed_dataset = SkeletonDataset(csv_file='Fall2_Cam5.avi_keys.csv',
                                          transform=None)

    print(transformed_dataset[0])

    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    for idx, sample in enumerate(train_loader):
        print(sample[0])
        print(sample[1])


if __name__ == "__main__":
    main()
