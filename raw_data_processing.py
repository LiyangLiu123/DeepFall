import os
import numpy as np
import random
import torch
from pip._vendor.distlib.compat import raw_input

path = '/Users/liuliyang/Downloads/nturgb+d_skeletons/'

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
            for j in range(0, num_bodies*27+1):
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
    # raw_input()
