import os
import numpy as np
import random
import csv
import torch
from pip._vendor.distlib.compat import raw_input

path = '/Users/liuliyang/Downloads/cs_train/'


data = []

list_of_files = os.listdir(path)
count_of_file = 0
for filename in list_of_files:
    # in case system files are read in
    if len(filename) is not 29:
        count_of_file += 1
        print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))
        continue

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
                numbers.append(list(map(float, lines[j].split(' ')[:3])))
            numbers = np.array(numbers)
            numbers = numbers.flatten()
            frames.append(numbers)

            for j in range(0, 28):
                lines = np.delete(lines, 0)
        else:
            # check which is main actor when having two skeletons detected
            numbers = []
            numbers1 = []
            numbers2 = []
            for j in range(3, 28):
                numbers1.append(list(map(float, lines[j].split(' ')[:3])))
            for j in range(30, 55):
                numbers2.append(list(map(float, lines[j].split(' ')[:3])))

            if sum(map(sum,numbers1)) >= sum(map(sum,numbers2)):
                numbers = numbers1
            else:
                numbers = numbers2
            numbers = np.array(numbers)
            numbers = numbers.flatten()

            frames.append(numbers)
            for j in range(0, 55):
                lines = np.delete(lines, 0)
        count += 1

    num_frames = len(frames)

    # choosing 20 frames out of 20 sub-sequences
    # skip if the file has fewer than 20 frames
    if num_frames < 20:
        count_of_file += 1
        print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))
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
    chosen_frames = np.array(chosen_frames)
    # chosen_frames = chosen_frames.reshape(chosen_frames.shape[0] * chosen_frames.shape[1], 1)
    chosen_frames = chosen_frames.flatten()
    with open('/Users/liuliyang/Downloads/cs_train_csv/data.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(chosen_frames)
    csvFile.close()

    count_of_file += 1
    print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))

    # raw_input()
