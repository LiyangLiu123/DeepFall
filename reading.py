from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random


class SkeletonDataset(Dataset):

    def __init__(self, path, transform=None):
        self.data = []
        for filename in os.listdir(path):
            file = open(path+filename, 'r')
            lines = file.readlines()
            num_frames = int(lines[0])
            lines = np.delete(lines, 0)

            # skip the files with multiple skeletons detected
            if int(lines[0]) is not 1:
                continue

            sub_length = int(num_frames / 20)
            idx = 0
            # 25 joints for each frame, each joint have 11 number
            packet = torch.Tensor(25, 11, 20)

            # divide the frames in 20 sub-sequences with almost the same length (sub_length+1 or sub_length)
            # and choose one frame randomly form each sub-sequence
            for i in range(20):
                if i < num_frames - 20 * sub_length:
                    chosen = random.randint(idx, idx+sub_length+1)
                    if int(lines[chosen*28+2]) is not 25:
                        print(filename)
                        print(chosen)
                        print(int(lines[chosen*28+2]))
                    # one frame contains 28 lines in the file
                    # start from the 3rd line of each frame
                    for j in range(25):
                        numbers = list(map(float, lines[chosen*28+3+j].split(' ')[:-1]))
                        if len(numbers) is not 11:
                            #print(filename)
                            #print(chosen)
                            #print(j)
                            #print(numbers)
                            #print(lines[chosen*28+3+j])
                            print("hehe")
                        packet[j, :, i] = torch.Tensor(numbers)
                    #print(packet)
                    #print(i)
                    idx += sub_length + 1
                else:
                    idx += sub_length

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
