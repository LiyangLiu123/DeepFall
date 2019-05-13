import numpy as np

test = np.load("test_label.npy")
test_len = len(test)*[30]

train = np.load("train_label.npy")
train_len = len(train)*[30]

np.save("test_len.npy", test_len)
np.save("train_len.npy", train_len)