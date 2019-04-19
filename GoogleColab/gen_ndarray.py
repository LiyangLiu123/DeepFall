import os
import numpy as np
import random
from pip._vendor.distlib.compat import raw_input

path = '/Users/liuliyang/Downloads/cv_train/'

list_of_files = os.listdir(path)
count_of_file = 0
targets_2cls = []
lens = []
data = []
for filename in list_of_files:
    #print(filename)
    # in case system files are read in
    if len(filename) is not 29:
        count_of_file += 1
        print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))
        continue

    target = int(filename[-11:-9]) - 1
    if target is not 42:
        x = random.randint(1, 1000)
        if x >= 17:
            # raw_input()
            count_of_file += 1
            print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))
            continue
        # raw_input()


    file = open(path + filename, 'r')
    lines = file.readlines()
    num_frames = int(lines[0])
    lines = np.delete(lines, 0)
    frames = []
    count = 0
    while count < num_frames:
        num_bodies = int(lines[0])
        if num_bodies >= 3:
            for _ in range(0, num_bodies * 27 + 1):
                lines = np.delete(lines, 0)
            count += 1
            continue
        if num_bodies is 0:
            lines = np.delete(lines, 0)
            count += 1
            continue
        if num_bodies is 1:
            numbers = []
            for j in range(3, 28):
                numbers.append(list(map(float, lines[j].split(' ')[:3])))
            for _ in range(25):
                numbers.append([0.0, 0.0, 0.0])
            frames.append(np.array(numbers))

            for j in range(0, 28):
                lines = np.delete(lines, 0)
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
        print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))
        continue

    lens.append(num_frames)

    # get the action class
    if int(filename[-11:-9]) is 43:
        target_2cls = 1
        # raw_input()
    else:
        target_2cls = 0
    targets_2cls.append(target_2cls)
    data.append(np.array(frames))
    #print(np.array(frames).shape)
    count_of_file += 1
    print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))
    #if count_of_file > 2000:
    #    break

    #raw_input()


np.save('/Users/liuliyang/Downloads/ndarray_cv/2cls/train_ntus_2cls.npy', data)
np.save('/Users/liuliyang/Downloads/ndarray_cv/2cls/train_ntus_2cls_len.npy', lens)
np.save('/Users/liuliyang/Downloads/ndarray_cv/2cls/train_ntus_2cls_label.npy', targets_2cls)

