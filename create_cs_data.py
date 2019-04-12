import os
from shutil import copyfile

path = '/Users/liuliyang/Downloads/nturgb+d_skeletons/'
train_path = '/Users/liuliyang/Downloads/cs_train/'
test_path = '/Users/liuliyang/Downloads/cs_test/'

filenames = os.listdir(path)
count = 0

for filename in filenames:
    subjectid = int(filename[10:12])
    if subjectid in [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]:
        copyfile(path + filename, train_path + filename)
    else:
        copyfile(path + filename, test_path + filename)
    count += 1
    print('{}/{} completed\n'.format(count, len(filenames)))
