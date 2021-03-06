import os
from shutil import copyfile

path = './nturgb+d_skeletons/'
train_path = './cv_train/'
test_path = './cv_test/'

os.mkdir(train_path)
os.mkdir(test_path)

filenames = os.listdir(path)
count = 0

for filename in filenames:
    camera_angle = int(filename[7:8])
    if camera_angle is not 1:
        copyfile(path + filename, train_path + filename)
    else:
        copyfile(path + filename, test_path + filename)
    count += 1
    print('create cv data: {}/{} completed'.format(count, len(filenames)))
