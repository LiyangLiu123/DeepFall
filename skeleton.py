"""Module using IndRNNCell to solve the skeleton task.
The hyper-parameters are taken from the paper as well.

"""
from indrnn import IndRNN
from indrnn import IndRNNv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from time import time
import random
import os

parser = argparse.ArgumentParser(description='PyTorch IndRNN NTU skeleton data')
# Default parameters taken from https://arxiv.org/abs/1803.04831
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--n-layer', type=int, default=4,
                    help='number of layer of IndRNN (default: 6)')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='number of hidden units in one IndRNN layer(default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-batch-norm', action='store_true', default=False,
                    help='disable frame-wise batch normalization after each layer')
parser.add_argument('--log_epoch', type=int, default=1,
                    help='after how many epochs to report performance')
parser.add_argument('--log_iteration', type=int, default=-1,
                    help='after how many iterations to report performance, deactivates with -1 (default: -1)')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='enable bidirectional processing')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 256)')
parser.add_argument('--max-steps', type=int, default=50000,
                    help='max iterations of training (default: 10000)')
parser.add_argument('--model', type=str, default="IndRNN",
                    help='if either IndRNN or LSTM cells should be used for optimization')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.batch_norm = not args.no_batch_norm

TIME_STEPS = 5500
# 20*25*11 landmarks
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)

cuda = torch.cuda.is_available()

# 0.25 for cross subject and 0.1 for cross view
dropout_prob = 0.25

cs_train_path = '/Users/liuliyang/Downloads/train/'
cs_test_path = '/Users/liuliyang/Downloads/test/'

cv_train_path = '/Users/liuliyang/Downloads/cv_train/'
cv_test_path = '/Users/liuliyang/Downloads/cv_test/'


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=2, model=IndRNN):
        super(Net, self).__init__()
        recurrent_inits = []
        for _ in range(n_layer - 1):
            recurrent_inits.append(
                lambda w: nn.init.uniform_(w, 0, RECURRENT_MAX)
            )
        recurrent_inits.append(lambda w: nn.init.uniform_(
            w, RECURRENT_MIN, RECURRENT_MAX))
        self.indrnn = model(
            input_size, hidden_size, n_layer, batch_norm=args.batch_norm,
            hidden_max_abs=RECURRENT_MAX, batch_first=True,
            bidirectional=args.bidirectional, recurrent_inits=recurrent_inits,
            gradient_clip=5, dropout_prob=dropout_prob
        )
        self.lin = nn.Linear(
            hidden_size * 2 if args.bidirectional else hidden_size, 10)
        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y, _ = self.indrnn(x, hidden)

        return self.lin(y[:, -1]).squeeze(1)


def main():
    # build model
    if args.model.lower() == "indrnn":
        model = Net(1, args.hidden_size, args.n_layer)
    elif args.model.lower() == "indrnnv2":
        model = Net(1, args.hidden_size, args.n_layer, IndRNNv2)
    else:
        raise Exception("unsupported cell model")

    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load data
    # train_data, test_data = sequential_MNIST(args.batch_size, cuda=cuda)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    transformed_dataset = SkeletonDataset(path=cs_train_path, transform=None)

    train_data = DataLoader(transformed_dataset,
                            batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    transformed_test_dataset = SkeletonDataset(path=cs_test_path, transform=None)

    test_data = DataLoader(transformed_test_dataset,
                           batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    # Train the model
    model.train()
    step = 0
    epochs = 0
    while step < args.max_steps:
        losses = []
        start = time()
        for data, target in train_data:
            # print(data.shape)
            # print(target)
            if cuda:
                data, target = data.cuda(), target.cuda()
            model.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().item())
            step += 1

            if step % args.log_iteration == 0 and args.log_iteration != -1:
                print(
                    "\tStep {} cross_entropy {}".format(
                        step, np.mean(losses)))
            if step >= args.max_steps:
                break
        if epochs % args.log_epoch == 0:
            print(
                "Epoch {} cross_entropy {} ({} sec.)".format(
                    epochs, np.mean(losses), time() - start))
        epochs += 1

    # get test error
    model.eval()
    correct = 0
    for data, target in test_data:
        if cuda:
            data, target = data.cuda(), target.cuda()
        out = model(data)
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print(
        "Test accuracy:: {:.4f}".format(
            100. * correct / len(test_data.dataset)))


class SkeletonDataset(Dataset):

    def __init__(self, path, transform=None):
        self.data = []

        list_of_files = os.listdir(path)
        count_of_file = 0
        for filename in list_of_files:
            # in case system files are read in
            if len(filename) is not 29:
                continue

            file = open(path + filename, 'r')
            lines = file.readlines()
            num_frames = int(lines[0])
            lines = np.delete(lines, 0)
            frames = []
            count = 0
            while count < num_frames:
                num_bodies = int(lines[0])
                if num_bodies is 1:
                    numbers = []
                    for j in range(3, 28):
                        numbers.append(list(map(float, lines[j].split(' ')[:-1])))
                    numbers = np.array(numbers)
                    numbers = numbers.flatten()
                    frames.append(numbers)
                    for j in range(0, 28):
                        lines = np.delete(lines, 0)
                else:
                    for j in range(0, num_bodies * 27 + 1):
                        lines = np.delete(lines, 0)
                count += 1
            num_frames = len(frames)

            # choosing 20 frames out of 20 sub-sequences
            # skip if the file has fewer than 20 frames
            if num_frames < 20:
                continue
            chosen_frames = []
            sub_length = int(num_frames / 20)
            idx = 0
            for i in range(20):
                if i < num_frames - 20 * sub_length:
                    chosen = random.randint(idx, idx + sub_length)
                    idx += sub_length + 1
                else:
                    chosen = random.randint(idx, idx + sub_length - 1)
                    idx += sub_length
                chosen_frames.append(frames[chosen])

            # get the action class
            if int(filename[-11:-9]) is 43:
                target = 1
            else:
                target = 0
            chosen_frames = np.array(chosen_frames)
            chosen_frames = chosen_frames.reshape(chosen_frames.shape[0] * chosen_frames.shape[1], 1)
            self.data.append([torch.Tensor(chosen_frames), target])
            count_of_file += 1
            print('{}/{} files reading completed'.format(count_of_file, len(list_of_files)))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    main()
