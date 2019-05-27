import numpy as np

seq_len=90

test = np.load("test_data_"+str(seq_len)+".npy")
test_len = []
for t in test:
	test_len.append(len(t))

train = np.load("train_data_"+str(seq_len)+".npy")
train_len = []
for y in train:
	train_len.append(len(y))

np.save('test_len_'+str(seq_len)+'.npy', test_len)
np.save('train_len_'+str(seq_len)+'.npy', train_len)
