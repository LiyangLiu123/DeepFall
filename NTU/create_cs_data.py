import os
from shutil import copyfile

path = './nturgb+d_skeletons/'
train_path = './cs_train/'
test_path = './cs_test/'

os.mkdir(train_path)
os.mkdir(test_path)

filenames = os.listdir(path)
count = 0

for filename in filenames:
    subjectid = int(filename[10:12])
    if subjectid in [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]:
        copyfile(path + filename, train_path + filename)
    else:
        copyfile(path + filename, test_path + filename)
    count += 1
    print('create cs data: {}/{} completed'.format(count, len(filenames)))
