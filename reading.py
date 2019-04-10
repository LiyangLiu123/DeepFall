from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random


class SkeletonDataset(Dataset):

    def __init__(self, path, transform=None):
        self.data = []
        data = []

        for filename in os.listdir(path):
            file = open(path + filename, 'r')
            lines = file.readlines()
            num_frames = int(lines[0])
            lines = np.delete(lines, 0)
            frames = []
            count = 0
            while count < num_frames:
                num_bodies = int(lines[0])
                if num_bodies is 1:
                    numbers = []
                    for j in range(3, 28):
                        numbers.append(list(map(float, lines[j].split(' ')[:-1])))
                    frames.append(numbers)
                    for j in range(0, 28):
                        lines = np.delete(lines, 0)
                else:
                    for j in range(0, num_bodies * 27 + 1):
                        lines = np.delete(lines, 0)
                count += 1
            num_frames = len(frames)

            # choosing 20 frames out of 20 sub-sequences
            # skip if the file has fewer than 20 frames
            if num_frames < 20:
                continue
            chosen_frames = []
            sub_length = int(num_frames / 20)
            idx = 0
            for i in range(20):
                if i < num_frames - 20 * sub_length:
                    chosen = random.randint(idx, idx + sub_length)
                    idx += sub_length + 1
                else:
                    chosen = random.randint(idx, idx + sub_length - 1)
                    idx += sub_length
                chosen_frames.append(frames[chosen])

            # get the action class
            if int(filename[-11:-9]) is 43:
                target = 1
            else:
                target = 0
            data.append([torch.Tensor(chosen_frames), target])
            print(filename)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    cuda = torch.cuda.is_available()
    batch_size = 100
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    transformed_dataset = SkeletonDataset(path='/Users/liuliyang/Downloads/nturgb+d_skeletons/',
                                          transform=None)

    print(transformed_dataset[0])

    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    for idx, sample in enumerate(train_loader):
        print(sample[0])
        print(sample[1])


if __name__ == "__main__":
    main()
