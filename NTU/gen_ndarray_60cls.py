import os
import numpy as np

path = './cs_train/'  # fill './cs_train', './cs_test', './cv_train' or './cv_test' here

list_of_files = os.listdir(path)
count_of_file = 0
targets = []
lens = []
data = []
for filename in list_of_files:
    # in case system files are read in
    if len(filename) is not 29:
        count_of_file += 1
        print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))
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
        print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))
        continue

    # get the action class
    targets.append(int(filename[-11:-9])-1)
    lens.append(num_frames)
    data.append(np.array(frames))
    count_of_file += 1
    print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))


np.save('./_60cls/'+path[2:-1]+'_60cls.npy', data)
np.save('./_60cls/'+path[2:-1]+'_60cls_len.npy', lens)
np.save('./_60cls/'+path[2:-1]+'_60cls_label.npy', targets)

