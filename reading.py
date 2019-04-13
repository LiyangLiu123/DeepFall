from __future__ import print_function, division
import torch
from pip._vendor.distlib.compat import raw_input
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import csv


class SkeletonDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data = []

        row_count = 0

        with open(csv_file, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                numbers = map(float, row)
                self.data.append(numbers)
                row_count += 1
                print("{} rows completed".format(row_count))

        csvFile.close()

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = np.array(self.data[idx][:-1])
        target = self.data[idx][-1]
        frames = frames.reshape(20, 75)

        sample = [torch.Tensor(frames), int(target)]

        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    cuda = torch.cuda.is_available()
    batch_size = 100
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    transformed_dataset = SkeletonDataset(csv_file='/Users/liuliyang/Downloads/csv/cs_train.csv',
                                          transform=None)

    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    for idx, sample in enumerate(train_loader):
        print("haha")
        #print(sample[1])


if __name__ == "__main__":
    main()
