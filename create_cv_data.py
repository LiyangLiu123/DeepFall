import os
from shutil import copyfile

path = '/Users/liuliyang/Downloads/nturgb+d_skeletons/'
train_path = '/Users/liuliyang/Downloads/cv_train/'
test_path = '/Users/liuliyang/Downloads/cv_test/'

filenames = os.listdir(path)
count = 0

for filename in filenames:
    camera_angle = int(filename[7:8])
    if camera_angle is not 3:
        copyfile(path + filename, train_path + filename)
    else:
        copyfile(path + filename, test_path + filename)
    count += 1
    print('{}/{} completed\n'.format(count, len(filenames)))