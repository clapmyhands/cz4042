import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import math

import visdom
import numpy as np

from collections import OrderedDict
from torchvision import datasets, transforms
from torch.autograd import Variable
from itertools import chain

def normalizeForImage(tensor: torch.FloatTensor):
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    return (tensor-min_val)/(max_val-min_val)

def visualizeLinearLayerWeight(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        vis = visdom.Visdom()
        weight_size = m.weight.size()
        side = math.floor(math.sqrt(weight_size[1]))

        idx = np.random.random_integers(0, weight_size[0]-1, 100)
        sample_weight = m.weight[idx, :]
        sample_weight = normalizeForImage(sample_weight)

        image = sample_weight.view(-1, 1, side, side).data.numpy()
        vis.images(image, nrow=5, opts=dict(
            title='Linear-[{}, ({}, {})]'.format(weight_size[0], side, side)
        ))

def initializeWeight(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)

def corruptInput(batch_input, corruption_level):
    corruption_matrix = np.random.binomial(1, 1-corruption_level, batch_input.size()[1:])
    corruption_matrix = torch.from_numpy(corruption_matrix).float()
    return batch_input*corruption_matrix


class VAENet(nn.Module):
    def __init__(self):
        super(VAENet, self).__init__()
        self.encodeLinear1 = nn.Linear(28*28, 900)
        self.encodeLinear2 = nn.Linear(900, 625)
        self.encodeLinear3 = nn.Linear(625, 400)
        self.decodeLinear3 = nn.Linear(400, 625)
        self.decodeLinear2 = nn.Linear(625, 900)
        self.decodeLinear1 = nn.Linear(900, 28*28)
        self.Sigmoid = nn.Sigmoid()
        self.activation = []

    def encode_layer(self, data, level):
        x = data.view(data.size()[0], -1)
        if level < 1 and level > 3:
            raise ValueError("Level should be 1, 2, or 3")
        if level == 1:
            output = self.Sigmoid(self.encodeLinear1(x))
            # self.activation.append(output)
        if level == 2:
            output = self.Sigmoid(self.encodeLinear2(x))
            # self.activation.append(output)
        if level == 3:
            output = self.Sigmoid(self.encodeLinear3(x))
        return output

    def encode(self, data):
        x = data
        x = self.encode_layer(x, 1)
        x = self.encode_layer(x, 2)
        x = self.encode_layer(x, 3)
        return x

    def decode_layer(self, latent_vector, level):
        x = latent_vector.view(latent_vector.size()[0], -1)
        if level < 1 and level > 3:
            raise ValueError("Level should be 1, 2, or 3")
        if level == 3:
            output = self.Sigmoid(self.decodeLinear3(x))
            # self.activation.append(output)
        if level == 2:
            output = self.Sigmoid(self.decodeLinear2(x))
            # self.activation.append(output)
        if level == 1:
            output = self.Sigmoid(self.decodeLinear1(x))
        return output

    def decode(self, latent_vector):
        x = latent_vector
        x = self.decode_layer(x, 3)
        x = self.decode_layer(x, 2)
        x = self.decode_layer(x, 1)
        return x

    def forward(self, data):
        x = data.view(data.size()[0], -1)
        latent_vector = self.encode(x)
        reconstruction = self.decode(latent_vector)
        return reconstruction.view(data.size())

    def forward_layer(self, data, level):
        if level < 1 and level > 3:
            raise ValueError("Level should be 1, 2, or 3")
        if level == 1:
            latent_vector = self.encode_layer(data, 1)
            reconstruction = self.decode_layer(latent_vector, 1)
            self.activation.append(latent_vector)
            self.activation.append(reconstruction)
        elif level == 2:
            latent_vector = self.encode_layer(data, 2)
            reconstruction = self.decode_layer(latent_vector, 2)
            self.activation.append(latent_vector)
            self.activation.append(reconstruction)
        elif level == 3:
            latent_vector = self.encode_layer(data, 3)
            reconstruction = self.decode_layer(latent_vector, 3)
            self.activation.append(latent_vector)
            self.activation.append(reconstruction)
        return reconstruction.view(data.size())

    def classify(self, data):
        x = data.view(data.size()[0], -1)
        x = self.encode(x)
        return x

    def clear_activation(self):
        self.activation = []


def SparsityCost(activation, penalty=0.5, sparsity=0.05):
    kl_cost = 0
    for i in activation:
        rhoj = torch.mean(i, 0, True)
        rho = Variable(torch.zeros(rhoj.size()[1]).fill_(sparsity))
        kl_cost += F.kl_div(torch.log(rhoj+1e-6), rho, size_average=False) +\
                   F.kl_div(torch.log(1-rhoj+1e-6), (1-rho), size_average=False)

    return kl_cost * penalty

def BceCriterion(output, target):
    cost = F.binary_cross_entropy(output, target, size_average=False)/output.size()[0]
    return cost

def main(classify=False, sparsity=False):
    vis = visdom.Visdom()
    learning_rate = 1e-1
    epochs = 3
    batch_size = 128
    corruption_level = 0.1
    momentum = 0
    if sparsity:
        momentum = 0.1

    vae = VAENet()
    vae.apply(initializeWeight)
    # criterion = torch.nn.BCELoss(size_average=False)
    criterion = BceCriterion

    if torch.cuda.is_available():
        vae = vae.cuda()
        criterion = criterion.cuda()

    train_dataset = datasets.MNIST("../data_mnists", train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST("../data_mnists", train=False, transform=transforms.ToTensor())
    # dataset = data.ConcatDataset((train_dataset, test_dataset))
    dataset = train_dataset
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("Sample size: {}".format(len(dataset)))
    print(vae)
    ### layer 1
    print("=========== Layer 1 ===========")
    parameter = chain(vae.encodeLinear1.parameters(), vae.decodeLinear1.parameters())
    optim = torch.optim.SGD(parameter, lr=learning_rate, momentum=momentum)
    iteration = 0
    for epoch in range(1, epochs+1):
        print("Epoch: {}".format(epoch))
        vae.train()
        for x, _ in loader:
            iteration += 1
            X = Variable(corruptInput(x, corruption_level))
            original = Variable(x)
            vae.zero_grad()

            reconstruction = vae.forward_layer(X, 1)
            loss = criterion(reconstruction, original)
            if sparsity:
                loss += SparsityCost(vae.activation)

            loss.backward()
            optim.step()
            vae.clear_activation()

            if iteration == 1:
                train_plot = vis.line(X=np.array([iteration]),
                                      Y=np.array([loss.data[0]]),
                                      opts=dict(
                                          title='Train Cost layer 1',
                                          xlabel='Iteration',
                                          ylabel='BCELoss'
                                      ))
            else:
                vis.updateTrace(X=np.array([iteration]),
                                Y=np.array([loss.data[0]]),
                                win=train_plot)

            # if iteration%1000==0:
            #     result = np.array([
            #         x[0].numpy(),
            #         X.data[0].numpy(),
            #         F.sigmoid(reconstruction).data[0].numpy()
            #     ])
            #     vis.images(result, opts=dict(caption=iteration))

    ### layer 2
    print("=========== Layer 2 ===========")
    parameter = chain(vae.encodeLinear2.parameters(), vae.decodeLinear2.parameters())
    optim = torch.optim.SGD(parameter, lr=learning_rate, momentum=momentum)
    iteration = 0
    for epoch in range(1, epochs + 1):
        print("Epoch: {}".format(epoch))
        vae.train()
        for x, _ in loader:
            iteration += 1
            # original = Variable(x)
            X = Variable(corruptInput(x, corruption_level))
            h = vae.encode_layer(X, 1)
            original = h
            vae.zero_grad()

            h = vae.forward_layer(h, 2)
            # h = vae.decode_layer(h, 1).view(original.size())
            reconstruction = h
            loss = criterion(reconstruction, original)
            if sparsity:
                loss += SparsityCost(vae.activation)

            loss.backward()
            optim.step()
            vae.clear_activation()

            if iteration == 1:
                train_plot = vis.line(X=np.array([iteration]),
                                      Y=np.array([loss.data[0]]),
                                      opts=dict(
                                          title='Train Cost layer 2',
                                          xlabel='Iteration',
                                          ylabel='BCELoss'
                                      ))
            else:
                vis.updateTrace(X=np.array([iteration]),
                                Y=np.array([loss.data[0]]),
                                win=train_plot)

    ### layer 3
    print("=========== Layer 3 ===========")
    parameter = chain(vae.encodeLinear3.parameters(), vae.decodeLinear3.parameters())
    optim = torch.optim.SGD(parameter, lr=learning_rate, momentum=momentum)
    iteration = 0
    for epoch in range(1, epochs + 1):
        print("Epoch: {}".format(epoch))
        vae.train()
        for x, _ in loader:
            iteration += 1
            # original = Variable(x)
            X = Variable(corruptInput(x, corruption_level))
            h = vae.encode_layer(X, 1)
            h = vae.encode_layer(h, 2)
            original = h
            vae.zero_grad()

            h = vae.forward_layer(h, 3)
            # h = vae.decode_layer(h, 2)
            # h = vae.decode_layer(h, 1).view(original.size())
            reconstruction = h
            loss = criterion(reconstruction, original)
            if sparsity:
                loss += SparsityCost(vae.activation)

            loss.backward()
            optim.step()
            vae.clear_activation()

            if iteration == 1:
                train_plot = vis.line(X=np.array([iteration]),
                                      Y=np.array([loss.data[0]]),
                                      opts=dict(
                                          title='Train Cost layer 3',
                                          xlabel='Iteration',
                                          ylabel='BCELoss'
                                      ))
            else:
                vis.updateTrace(X=np.array([iteration]),
                                Y=np.array([loss.data[0]]),
                                win=train_plot)

    # Visualize Weights
    vae.apply(visualizeLinearLayerWeight)

    # Visualize Activation
    vae.eval()
    test_loader = data.DataLoader(test_dataset, shuffle=True, batch_size=100)
    for x, _ in test_loader:
        X = Variable(x)
        reconstruction = vae(X)
        images = torch.cat([X.data, reconstruction.data], 3).numpy()
        vis.images(images, nrow=5, opts=dict(title='original-reconstruction'))
        X = X.view(X.size()[0], -1)
        # encode 1 (784 -> 900)
        X = vae.encode_layer(X, 1)
        image_x = X.view(-1, 1, 30, 30)
        vis.images(image_x.data.numpy(), nrow=5, opts=dict(title='Sigmoid-EncodeLinear-1'))
        # encode 2 (900 -> 625)
        X = vae.encode_layer(X, 2)
        image_x = X.view(-1, 1, 25, 25)
        vis.images(image_x.data.numpy(), nrow=5, opts=dict(title='Sigmoid-EncodeLinear-2'))
        # encode 3 (625 -> 400)
        X = vae.encode_layer(X, 3)
        image_x = X.view(-1, 1, 20, 20)
        vis.images(image_x.data.numpy(), nrow=5, opts=dict(title='Sigmoid-EncodeLinear-3'))
        # decode 3 (400 -> 625)
        X = vae.decode_layer(X, 3)
        image_x = X.view(-1, 1, 25, 25)
        vis.images(image_x.data.numpy(), nrow=5, opts=dict(title='Sigmoid-DecodeLinear-3'))
        # decode 2 (625 -> 900)
        X = vae.decode_layer(X, 2)
        image_x = X.view(-1, 1, 30, 30)
        vis.images(image_x.data.numpy(), nrow=5, opts=dict(title='Sigmoid-DecodeLinear-2'))
        break

    if classify:
        parameter = chain(vae.encodeLinear1.parameters(),
                          vae.encodeLinear2.parameters(),
                          vae.encodeLinear3.parameters())
        optim = torch.optim.SGD(parameter, lr=learning_rate)
        vae.train()
        classifier_loss = nn.CrossEntropyLoss()
        iteration = 0
        for epoch in range(1,epochs+1):
            for x, y in loader:
                iteration += 1
                vae.zero_grad()
                X = Variable(x)
                Y = Variable(y)

                pred = vae.classify(X)
                loss = classifier_loss(pred, Y)

                loss.backward()
                optim.step()

                accuracy = 0.0
                for x_test, y_test in test_loader:
                    X_test = Variable(x_test)
                    _, pred = torch.max(vae.classify(X_test), 1)
                    accuracy += torch.sum(pred.data == y_test)
                accuracy = 100 * accuracy / len(test_dataset)

                #plot loss and accuracy here
                if iteration == 1:
                    softmax_train_plot = vis.line(X=np.array([iteration]),
                                                  Y=np.array([loss.data[0]]),
                                                  opts=dict(
                                                      title='Softmax Train Cost',
                                                      xlabel='Iteration',
                                                      ylabel='CrossEntropyLoss'
                                                  ))

                    softmax_test_plot = vis.line(X=np.array([iteration]),
                                                 Y=np.array([accuracy]),
                                                 opts=dict(
                                                     title='Softmax Test Accuracy',
                                                     xlabel='Iteration',
                                                     ylabel='% Accuracy'
                                                 ))
                else:
                    vis.updateTrace(X=np.array([iteration]),
                                    Y=np.array([loss.data[0]]),
                                    win=softmax_train_plot)
                    vis.updateTrace(X=np.array([iteration]),
                                    Y=np.array([accuracy]),
                                    win=softmax_test_plot)

if __name__ == '__main__':
    main(classify=False, sparsity=True)