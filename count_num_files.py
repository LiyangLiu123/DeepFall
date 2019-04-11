import os

path = '/Users/liuliyang/Downloads/nturgb+d_skeletons/'

angles = [0, 0, 0]
subjects = [0] * 40

for filename in os.listdir(path):
    angles[int(filename[7:8])-1] += 1
    subjects[int(filename[10:12])-1] += 1

file = open('samples_with_missing_data')
lines = file.readlines()

for line in lines:
    angles[int(line[7:8])-1] += 1
    subjects[int(line[10:12])-1] += 1

print(angles)
print(subjects)

sum = 0
for i in range(19, 40):
    sum += subjects[i]
print(sum)
