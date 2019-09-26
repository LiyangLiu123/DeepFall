import os

file = open("samples_with_missing_data")

lines = file.readlines()

names = []

for line in lines:
    names.append(line[:-1])

for name in names:
    os.remove("./nturgb+d_skeletons/"+name+".skeleton")
