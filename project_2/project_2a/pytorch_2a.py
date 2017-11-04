import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np

import visdom

from torchvision import datasets, transforms
from torch.autograd import Variable

def initializeWeight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight)
        nn.init.constant(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
        nn.init.constant(m.bias, 0)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # ?, 1, 28, 28
        self.Conv1 = nn.Conv2d(1, 15, 9, 1, 0)
        initializeWeight(self.Conv1)
        self.Relu1 = nn.ReLU()
        # ?, 15, 20, 20
        self.MaxPool1 = nn.MaxPool2d(2)
        # ?, 15, 10, 10
        self.Conv2 = nn.Conv2d(15, 20, 5, 1, 0)
        initializeWeight(self.Conv2)
        self.Relu2 = nn.ReLU()
        # ?, 20, 6, 6
        self.MaxPool2 = nn.MaxPool2d(2)
        # ?, 20, 3, 3
        self.Fc1 = nn.Linear(180, 100)
        initializeWeight(self.Fc1)
        self.Relu3 = nn.ReLU()
        self.Fc2 = nn.Linear(100, 10)
        initializeWeight(self.Fc2)


    def forward(self, data):
        x = self.Conv1(data)
        x = self.Relu1(x)
        x = self.MaxPool1(x)
        x = self.Conv2(x)
        x = self.Relu2(x)
        x = self.MaxPool2(x)
        x = x.view(-1, 180)
        x = self.Fc1(x)
        x = self.Relu3(x)
        x = self.Fc2(x)
        return x

def train(net, criterion, optimizer, train_set):
    train_size, train_loader = train_set
    net.train()
    cost = 0.0
    for idx, (x, y) in enumerate(train_loader):
        X, target = Variable(x), Variable(y)
        if torch.cuda.is_available():
            X, target = X.cuda(), target.cuda()
        net.zero_grad()

        pred = net(X)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        cost += loss.data[0] * len(x)
    cost = cost / train_size
    return cost

def test(net, test_set):
    test_size, test_loader = test_set
    net.eval()
    accuracy = 0
    for _, (x, y) in enumerate(test_loader):
        X, target = Variable(x), y
        if torch.cuda.is_available():
            X = X.cuda()

        output = net(X)
        _, pred = torch.max(output.data, 1)
        accuracy += (pred == target).sum()
    accuracy = 100 * accuracy / test_size
    return accuracy

def main():
    learning_rate = 0.05
    decay = 1e-6
    batch_size = 128
    epochs = 100
    # question 1
    momentum = 0
    # question 2
    # momentum = 0.5

    train_dataset = datasets.MNIST("../data_mnists", train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST("../data_mnists", train=False, transform=transforms.ToTensor())
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=True, num_workers=4)
    train_set = (train_size, train_loader)
    test_set =  (test_size, test_loader)

    vis = visdom.Visdom()
    win = None

    net = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=decay, momentum=momentum)

    # learning_rate = 1e-3
    # decay = 1e-4
    # alpha = 0.9
    # epsilon = 1e-6
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, alpha=alpha, weight_decay=decay, eps=epsilon)

    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()

    print(net)


    train_cost = []
    test_accuracy = []
    for epoch in range(epochs):
        cost = train(net, criterion, optimizer, train_set)
        accuracy = test(net, test_set)

        train_cost.append(cost)
        test_accuracy.append(accuracy)

        if epoch == 0:
            train_plot = vis.line(X=np.array([epoch+1]), Y=np.array([cost]), opts=dict(
                title='Training Cost',
                xlabel='iteration',
                ylabel='CrossEntropyLoss'
            ))
            test_plot = vis.line(X=np.array([epoch+1]), Y=np.array([accuracy]), opts=dict(
                title='Test Accuracy',
                xlabel='iteration',
                ylabel='Accuracy'
            ))
        else:
            vis.updateTrace(X=np.array([epoch+1]), Y=np.array([cost]), win=train_plot)
            vis.updateTrace(X=np.array([epoch+1]), Y=np.array([accuracy]), win=test_plot)

    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    count = 0
    for i, _ in test_loader:
        temp = Variable(i)
        temp1 = net.Relu1(net.Conv1(temp))
        temp2 = net.MaxPool1(temp1)
        temp3 = net.Relu2(net.Conv2(temp2))
        temp4 = net.MaxPool2(temp3)
        vis.image(F.upsample(temp, scale_factor=6).data.numpy(), opts=dict(caption='Sample-{}'.format(count), jpgquality=100))
        vis.images(F.upsample(temp1.view(15, 1, 20, 20), scale_factor=6).data.numpy(), nrow=5, opts=dict(caption='Conv1-{}'.format(count), jpgquality=100))
        vis.images(F.upsample(temp2.view(15, 1, 10, 10), scale_factor=6).data.numpy(), nrow=5, opts=dict(caption='MaxPool1-{}'.format(count), jpgquality=100))
        vis.images(F.upsample(temp3.view(20, 1, 6, 6), scale_factor=6).data.numpy(), nrow=5, opts=dict(caption='Conv2-{}'.format(count), jpgquality=100))
        vis.images(F.upsample(temp4.view(20, 1, 3, 3), scale_factor=6).data.numpy(), nrow=5, opts=dict(caption='MaxPool2-{}'.format(count), jpgquality=100))
        count += 1
        if count == 2:
            break


if __name__ == '__main__':
    main()
