import os
from shutil import copyfile

path = '/Users/liuliyang/Downloads/nturgb+d_skeletons/'
train_path = '/Users/liuliyang/Downloads/cs_train/'
test_path = '/Users/liuliyang/Downloads/cs_test/'

filenames = os.listdir(path)
count = 0

for filename in filenames:
    subjectid = int(filename[10:12])
    if subjectid is not 4 and subjectid < 20:
        copyfile(path + filename, train_path + filename)
    else:
        copyfile(path + filename, test_path + filename)
    count += 1
    print('{}/{} completed\n'.format(count, len(filenames)))
