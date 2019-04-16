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
import csv

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
parser.add_argument('--max-steps', type=int, default=10000,
                    help='max iterations of training (default: 10000)')
parser.add_argument('--model', type=str, default="IndRNN",
                    help='if either IndRNN or LSTM cells should be used for optimization')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.batch_norm = not args.no_batch_norm

TIME_STEPS = 20
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)

cuda = args.cuda

# 0.25 for cross subject and 0.1 for cross view
dropout_prob = 0.25

cs_train_csv = '/Users/liuliyang/Downloads/csv/cs_train.csv'
cs_test_csv = '/Users/liuliyang/Downloads/csv/cs_test.csv'

cv_train_csv = '/Users/liuliyang/Downloads/csv/cv_train.csv'
cv_test_csv = '/Users/liuliyang/Downloads/csv/cv_test.csv'


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
            hidden_size * 2 if args.bidirectional else hidden_size, 60)
        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y, _ = self.indrnn(x, hidden)

        return self.lin(y[:, -1]).squeeze(1)


def main():
    # build model
    if args.model.lower() == "indrnn":
        model = Net(75, args.hidden_size, args.n_layer)
    elif args.model.lower() == "indrnnv2":
        model = Net(75, args.hidden_size, args.n_layer, IndRNNv2)
    else:
        raise Exception("unsupported cell model")

    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}

    transformed_dataset = SkeletonDataset(csv_file=cs_train_csv, transform=None)

    train_data = DataLoader(transformed_dataset,
                            batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    transformed_test_dataset = SkeletonDataset(csv_file=cs_test_csv, transform=None)

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
            model.train()
        epochs += 1


class SkeletonDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data = []

        row_count = 0

        with open(csv_file, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                numbers = map(float, row)
                self.data.append(numbers)
                row_count += 1
                print("{} rows completed".format(row_count))

        csvFile.close()

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = np.array(self.data[idx][:-1])
        target = self.data[idx][-1]
        frames = frames.reshape(20, 75)

        sample = [torch.Tensor(frames), int(target)-1]

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    main()
