import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate npy')
parser.add_argument('path', help='the path to generate npy')

args = parser.parse_args()

path = args.path  # fill './cs_train', './cs_test', './cv_train' or './cv_test' here

upsample_ratio = 25

list_of_files = os.listdir(path)
count_of_file = 0
targets_2cls = []
lens = []
data = []
for filename in list_of_files:
    # in case system files are read in
    if len(filename) is not 29:
        count_of_file += 1
        print('gen_2cls '+path+' : {}/{} files reading completed'.format(count_of_file, len(list_of_files)))
        continue

    target = int(filename[-11:-9]) - 1

    file = open(path + filename, 'r')
    lines = file.readlines()
    num_frames = int(lines[0])
    lines = np.delete(lines, 0)
    frames = []
    count = 0
    while count < num_frames:
        num_bodies = int(lines[0])
        # more than 2 bodies
        if num_bodies >= 3:
            for _ in range(0, num_bodies * 27 + 1):
                lines = np.delete(lines, 0)
            count += 1
            continue
        # zero body
        elif num_bodies == 0:
            lines = np.delete(lines, 0)
            count += 1
            continue
        # 1 body
        elif num_bodies == 1:
            numbers = []
            for j in range(3, 28):
                numbers.append(list(map(float, lines[j].split(' ')[:3])))
            # the second body is all 0
            for _ in range(25):
                numbers.append([0.0, 0.0, 0.0])
            frames.append(np.array(numbers))

            for j in range(0, 28):
                lines = np.delete(lines, 0)
        # 2 bodies
        else:
            numbers = []
            for j in range(3, 28):
                numbers.append(list(map(float, lines[j].split(' ')[:3])))
            for j in range(30, 55):
                numbers.append(list(map(float, lines[j].split(' ')[:3])))
            frames.append(np.array(numbers))
            for j in range(0, num_bodies*27+1):
                lines = np.delete(lines, 0)
        count += 1

    num_frames = len(frames)

    if num_frames is 0:
        count_of_file += 1
        print('gen_2cls '+path+' : {}/{} files reading completed'.format(count_of_file, len(list_of_files)))
        continue

    # get the action class
    if int(filename[-11:-9]) is 43:
        target_2cls = 1
        for _ in range(upsample_ratio):
            targets_2cls.append(target_2cls)
            lens.append(num_frames)
            data.append(np.array(frames))
    else:
        target_2cls = 0
        targets_2cls.append(target_2cls)
        lens.append(num_frames)
        data.append(np.array(frames))
    count_of_file += 1
    print('gen_2cls '+path+' : {}/{} files reading completed'.format(count_of_file, len(list_of_files)))

os.mkdir("./up_sampling_2cls")

np.save('./up_sampling_2cls/'+path[2:-1]+'_2cls.npy', data)
np.save('./up_sampling_2cls/'+path[2:-1]+'_2cls_len.npy', lens)
np.save('./up_sampling_2cls/'+path[2:-1]+'_2cls_label.npy', targets_2cls)

