import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


def scale(x, min=0, max=1):
    epsilon = 1e-8
    x_min , _ = torch.min(x, 0)
    x_max , _ = torch.max(x, 0)
    x_min = x_min.expand_as(x)
    x_max = x_max.expand_as(x)
    # avoid div by 0
    temp_x = torch.div((x-x_min),(x_max-x_min+epsilon))
    return temp_x * (max-min) + min


class simple_softmax(nn.Module):
    def __init__(self):
        super(simple_softmax, self).__init__()

        self.line1 = nn.Linear(36,10)
        nn.init.xavier_uniform(self.line1.weight)
        nn.init.constant(self.line1.bias,0)

        self.sigmoid = nn.Sigmoid()

        self.line2 = nn.Linear(10,6)
        nn.init.xavier_uniform(self.line2.weight)
        nn.init.constant(self.line2.bias,0)

        self.logSoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.sigmoid(self.line1(x))
        return self.logSoftmax(self.line2(x))


def time_from_start(startTime, msg):
    print("Time from start: {:.10f} msg: {}".format(time.time()-startTime, msg))


def load_landsat_dataset(filename, delimiter=' '):
    data = torch.from_numpy(np.loadtxt(filename, delimiter=delimiter))
    size = len(data)
    x, y = data[:, :-1].type(torch.FloatTensor), data[:, -1].type(torch.LongTensor)
    y[y == 7] = 6
    y = y - 1  # change to 0 index
    dataset = torch.utils.data.TensorDataset(scale(x), y)
    return size, dataset


def train_landsat(train, test, learning_rate, decay, epochs, batch_size):
    """
    :param train: {train_size : int, train_dataset : torch.utils.data.Dataset}
    :param test: {test_size, test_dataset}
    :param learning_rate:
    :param decay:
    :param epochs:
    :param batch_size:
    :return:
    """
    train_size, train_dataset = train
    test_size, test_dataset = test

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = simple_softmax()
    optimizer = optim.SGD(model.parameters(), learning_rate, weight_decay=decay)
    nllloss = nn.NLLLoss()

    train_cost = []
    test_accuracy = []
    avg_weight_update_time = 0.0
    print("batch size: {}".format(batch_size))
    start_time = time.time()
    for i in range(1, epochs + 1):
        cost = 0.0
        weight_update_start = time.time()
        for _, (X, Y) in enumerate(train_loader):
            x, y = Variable(X), Variable(Y)
            optimizer.zero_grad()
            # output is log softmax, use nll loss for cross entropy
            output = model(x)
            loss = nllloss(output, y)
            cost += loss

            loss.backward()
            optimizer.step()
        train_cost.append((cost / train_size).data.numpy())
        weight_update_end = time.time()
        avg_weight_update_time += weight_update_end - weight_update_start

        correct = 0.0
        for X, Y in test_loader:
            x, y = Variable(X), Y
            # torch.max returns value then position(argmax)
            _, output = torch.max(model(x), 1)
            correct += torch.sum(y == output.data)
        test_accuracy.append((correct / test_size))

        if i % 100 == 0:
            time_from_start(start_time, "epoch: {}".format(i))

    avg_weight_update_time /= epochs
    return train_cost, test_accuracy, avg_weight_update_time

if __name__ == "__main__":
    decay=1e-6
    learning_rate = 0.01
    epochs = 500
    batch_sizes = [4,8,16,32,64]

    train = load_landsat_dataset('sat_train.txt')
    test = load_landsat_dataset('sat_test.txt')

    train_costs = []
    test_accuracies = []
    avg_weight_update_times = []
    for batch_size in batch_sizes:
        train_cost, test_accuracy, avg_weight_update_time= train_landsat(train, test, learning_rate, decay, epochs, batch_size)
        train_costs.append(train_cost)
        test_accuracies.append(test_accuracy)
        avg_weight_update_times.append(avg_weight_update_time)

    print(avg_weight_update_times)
    # Plots
    plt.figure()
    for train_cost in train_costs:
        plt.plot(range(epochs), train_cost)
    plt.legend(batch_sizes, loc='upper right')
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    plt.savefig('p1a_sample_cost.png')

    plt.figure()
    for test_accuracy in test_accuracies:
        plt.plot(range(epochs), test_accuracy)
    plt.legend(batch_sizes, loc='lower right')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.savefig('p1a_sample_accuracy.png')

    plt.show()