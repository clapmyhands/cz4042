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
    x_min, _ = torch.min(x, 0)
    x_max, _ = torch.max(x, 0)
    x_min = x_min.expand_as(x)
    x_max = x_max.expand_as(x)
    # avoid div by 0
    temp_x = torch.div((x-x_min), (x_max-x_min+epsilon))
    return temp_x * (max-min) + min

def init_weights(module):
    classname = module.__class__.__name__
    if(classname.find('Linear') != -1):
        nn.init.xavier_uniform(module.weight)
        nn.init.constant(module.bias, 0)

class simple_softmax(nn.Module):
    """
        Design a 3 layer Feed-Forward model by default with size 36x10x6
    """
    def __init__(self, ):
        super(simple_softmax, self).__init__()
        self.main = nn.Sequential(*self.__design_model())

    def forward(self, x):
        x = self.main(x)
        return x

    def __design_model(self, layers_size=None):
        if(layers_size is None):
            layers_size = [36, 10, 6]
        if(type(layers_size)!=list):
            raise ValueError
        layers = []
        for iter in range(len(layers_size)-1):
            if (iter != 0):
                layers.append(nn.Sigmoid())
            layers.append(nn.Linear(*layers_size[iter: iter+2]))
        layers.append(nn.LogSoftmax())
        return layers

    def build_model(self, layers_size):
        model_design = self.__design_model(layers_size)
        self.main = nn.Sequential(*model_design)


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


def train_landsat(train, test, learning_rate, decay, epochs, batch_size=16, layers_size=None):
    """
    :param train: {train_size : int, train_dataset : torch.utils.data.Dataset}
    :param test: {test_size, test_dataset}
    :param learning_rate:
    :param decay:
    :param epochs:
    :param batch_size:
    :return:
    """
    torch.manual_seed(10)
    train_size, train_dataset = train
    test_size, test_dataset = test

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = simple_softmax()
    # handle more than 3 layer and with different sizes
    if(layers_size is not None):
        model.build_model(layers_size)
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), learning_rate, weight_decay=decay)
    nllloss = nn.NLLLoss()
    print(model)

    train_cost = []
    test_accuracy = []
    avg_weight_update_time = 0.0

    # initial cost
    cost = 0.0
    for _, (X, Y) in enumerate(train_loader):
        x, y = Variable(X), Variable(Y)
        output = model(x)
        loss = nllloss(output, y)
        cost += loss * len(x)
    train_cost.append((cost / train_size).data.numpy())

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
            # NLLLoss from pytorch returns average per batch. multiply with size to get cost per batch.
            cost += loss * len(x)

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


def train_tune_batchsizes(train, test, batch_sizes, learning_rate=1e-2, decay=1e-6, epochs=500):
    train_costs = []
    test_accuracies = []
    avg_weight_update_times = []
    for batch_size in batch_sizes:
        print("batch size: {}".format(batch_size))
        train_cost, test_accuracy, avg_weight_update_time = \
            train_landsat(train, test, learning_rate, decay, epochs, batch_size)
        train_costs.append(train_cost)
        test_accuracies.append(test_accuracy)
        avg_weight_update_times.append(avg_weight_update_time)

    print(avg_weight_update_times)
    # Plots
    plt.figure()
    for train_cost in train_costs:
        plt.plot(range(epochs + 1), train_cost)
    plt.legend(batch_sizes, loc='upper right')
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    plt.savefig('pytorch_1a_feedforward_training_cost.png')

    plt.figure()
    for test_accuracy in test_accuracies:
        plt.plot(range(epochs), test_accuracy)
    plt.legend(batch_sizes, loc='lower right')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.savefig('pytorch_1a_feedforward_test_accuracy.png')

    plt.figure()
    plt.plot(batch_sizes, avg_weight_update_times)
    plt.xticks(batch_sizes)
    plt.xlabel('batch size')
    plt.ylabel('avg update time')
    plt.title('update time for batch size')
    plt.savefig('pytorch_1a_feedforward_avg_update_time.png')
    plt.show()


def train_tune_layersizes(train, test, layers_sizes, batch_size, learning_rate=1e-2, decay=1e-6, epochs=500):
    train_costs = []
    test_accuracies = []
    avg_weight_update_times = []
    print("batch size: {}".format(batch_size))
    for layers_size in layers_sizes:
        print("layers size: {}".format(layers_size))
        train_cost, test_accuracy, avg_weight_update_time = \
            train_landsat(train, test, learning_rate, decay, epochs, batch_size, layers_size)
        train_costs.append(train_cost)
        test_accuracies.append(test_accuracy)
        avg_weight_update_times.append(avg_weight_update_time)

    print(avg_weight_update_times)

    layers_sizes_string = ['x'.join(str(i) for i in layers_size) for layers_size in layers_sizes]
    # Plots
    plt.figure()
    for train_cost in train_costs:
        plt.plot(range(epochs + 1), train_cost)
    plt.legend(layers_sizes_string, loc='upper right')
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    plt.savefig('pytorch_1a_layerssizes_training_cost.png')

    plt.figure()
    for test_accuracy in test_accuracies:
        plt.plot(range(epochs), test_accuracy)
    plt.legend(layers_sizes_string, loc='lower right')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.savefig('pytorch_1a_layerssize_test_accuracy.png')

    plt.figure()
    plt.plot(range(len(layers_sizes_string)), avg_weight_update_times)
    plt.xticks(range(len(layers_sizes_string)), layers_sizes_string)
    plt.xlabel('layers_sizes')
    plt.ylabel('avg update time')
    plt.title('update time for layer size')
    plt.savefig('pytorch_1a_layerssize_avg_update_time.png')

    plt.figure()
    for train_cost in train_costs:
        plt.plot(range(50), train_cost[-50:])
    plt.legend(layers_sizes_string, loc='upper right')
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('last 50 epoch training cost')
    plt.savefig('pytorch_1a_layerssizes_last50_training_cost.png')

    plt.figure()
    for test_accuracy in test_accuracies:
        plt.plot(range(50), test_accuracy[-50:])
    plt.legend(layers_sizes_string, loc='lower right')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('last 50 epoch test accuracy')
    plt.savefig('pytorch_1a_layerssizes_last50_test_accuracy.png')
    # plt.show()


def train_tune_decay(train, test, layers_size, batch_size, decays, learning_rate=1e-2, epochs=500):
    train_costs = []
    test_accuracies = []
    avg_weight_update_times = []
    print("batch size: {}".format(batch_size))
    for decay in decays:
        print("layers size: {}".format(layers_size))
        train_cost, test_accuracy, avg_weight_update_time = \
            train_landsat(train, test, learning_rate, decay, epochs, batch_size, layers_size)
        train_costs.append(train_cost)
        test_accuracies.append(test_accuracy)
        avg_weight_update_times.append(avg_weight_update_time)
    print(avg_weight_update_times)

    plt.figure()
    for train_cost in train_costs:
        plt.plot(range(epochs + 1), train_cost)
    plt.legend(decays, loc='upper right')
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    plt.savefig('pytorch_1a_decay_training_cost.png')

    plt.figure()
    for test_accuracy in test_accuracies:
        plt.plot(range(epochs), test_accuracy)
    plt.legend(decays, loc='lower right')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.savefig('pytorch_1a_decay_test_accuracy.png')


if __name__ == "__main__":
    learning_rate = 0.01
    epochs = 500
    train = load_landsat_dataset('sat_train.txt')
    test = load_landsat_dataset('sat_test.txt')

    # # question 1
    # batch_sizes = [32]
    # decay = 1e-6
    # train_tune_batchsizes(train, test, batch_sizes, learning_rate, decay, epochs)

    # # question 2
    # batch_sizes = [4, 8, 16, 32, 64]
    # decay = 1e-6
    # train_tune_batchsizes(train, test, batch_sizes, learning_rate, decay, epochs)

    # # question 3
    # batch_size = 16
    # layers_sizes = [
    #     [36,5,6],
    #     [36,10,6],
    #     [36,15,6],
    #     [36,20,6],
    #     [36,25,6]
    # ]
    # decay = 1e-6
    # train_tune_layersizes(train, test, layers_sizes, batch_size, learning_rate, decay, epochs)

    # question 4
    batch_size = 16
    layers_size = [36,10,6]
    decays = [0, 1e-3, 1e-6, 1e-9, 1e-12]
    train_tune_decay(train, test, layers_size, batch_size, decays, learning_rate, epochs)


    # # question 5
    # layers_sizes = [
    #     [36,10,6],
    #     [36,10,10,6]
    # ]
    # batch_size = 32
    # decay = 1e-6
    # train_tune_layersizes(train, test, layers_sizes, batch_size, learning_rate, decay, epochs)