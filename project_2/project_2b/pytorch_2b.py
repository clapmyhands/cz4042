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
    batch_size = tensor.size()[0]
    max_val, _ = torch.max(tensor.view(batch_size, -1), 1, keepdim=True)
    min_val, _ = torch.min(tensor.view(batch_size, -1), 1, keepdim=True)
    return (tensor-min_val)/(max_val-min_val+1e-12), max_val, min_val

def visualizeLinearLayerWeight(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        vis = visdom.Visdom()
        weight_size = m.weight.size()
        side = math.floor(math.sqrt(weight_size[1]))

        idx = np.random.random_integers(0, weight_size[0]-1, 100)
        idx = torch.from_numpy(idx).long()
        if torch.cuda.is_available():
            idx = idx.cuda()
        sample_weight = m.weight[idx, :]
        sample_weight, max_val, min_val = normalizeForImage(sample_weight)

        print("{}: {} x {}".format(classname, *weight_size))
        # for mini, maxi in zip(min_val.data.numpy(), max_val.data.numpy()):
        #     print(mini, " ", maxi)

        image = sample_weight.view(-1, 1, side, side).cpu().data.numpy()
        vis.images(image, nrow=5, opts=dict(
            title='Linear-[{}, ({}, {})]'.format(weight_size[0], side, side)
        ))

def initializeWeight(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
        nn.init.constant(m.bias, 0)

def corruptInput(batch_input, corruption_level):
    corruption_matrix = np.random.binomial(1, 1-corruption_level, batch_input.size()[1:])
    corruption_matrix = torch.from_numpy(corruption_matrix).float()
    if torch.cuda.is_available():
        corruption_matrix = corruption_matrix.cuda()
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
        self.classify_out  = nn.Linear(400, 10)
        self.Sigmoid = nn.Sigmoid()
        self.activation = []

    def encode_layer(self, data, level):
        x = data.view(data.size()[0], -1)
        if level < 1 and level > 3:
            raise ValueError("Level should be 1, 2, or 3")
        if level == 1:
            output = self.Sigmoid(self.encodeLinear1(x))
        if level == 2:
            output = self.Sigmoid(self.encodeLinear2(x))
        if level == 3:
            output = self.Sigmoid(self.encodeLinear3(x))
            # output = self.encodeLinear3(x)
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
        if level == 2:
            output = self.Sigmoid(self.decodeLinear2(x))
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
            # self.activation.append(reconstruction)
        elif level == 2:
            latent_vector = self.encode_layer(data, 2)
            reconstruction = self.decode_layer(latent_vector, 2)
            self.activation.append(latent_vector)
            self.activation.append(reconstruction)
        elif level == 3:
            latent_vector = self.encode_layer(data, 3)
            reconstruction = self.decode_layer(latent_vector, 3)
            # self.activation.append(latent_vector)
            self.activation.append(reconstruction)
        return reconstruction.view(data.size())

    def classify(self, data):
        x = data.view(data.size()[0], -1)
        x = self.encode(x)
        x = self.classify_out(x)
        return x

    def clear_activation(self):
        self.activation = []


def SparsityCost(activation, penalty=0.5, sparsity=0.05):
    kl_cost = 0
    for i in activation:
        rhoj = torch.mean(i, 0, True)
        rho = Variable(torch.zeros(rhoj.size()[1]).fill_(sparsity))
        if torch.cuda.is_available():
            rhoj, rho= rhoj.cuda(), rho.cuda()
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

    if torch.cuda.is_available():
        vae = vae.cuda()

    ### load datasets
    train_dataset = datasets.MNIST("../data_mnists", train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST("../data_mnists", train=False, transform=transforms.ToTensor())
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("Train dataset: {} samples".format(len(train_dataset)))
    print("Test dataset: {} samples".format(len(test_dataset)))
    print(vae)

    ### layer 1
    print("=========== Layer 1 ===========")
    parameter = chain(vae.encodeLinear1.parameters(), vae.decodeLinear1.parameters())
    optim = torch.optim.SGD(parameter, lr=learning_rate, momentum=momentum)
    iteration = 0
    for epoch in range(1, epochs+1):
        print("Epoch: {}".format(epoch))
        vae.train()
        cost = 0.0
        iteration += 1
        for x, _ in train_loader:
            if torch.cuda.is_available():
                x = x.cuda()
            X = Variable(corruptInput(x, corruption_level))
            original = Variable(x)
            vae.zero_grad()

            reconstruction = vae.forward_layer(X, 1)
            loss = BceCriterion(reconstruction, original)
            if sparsity:
                loss += SparsityCost(vae.activation)

            loss.backward()
            optim.step()
            vae.clear_activation()
            cost += loss.data[0] * len(x)
        cost /= len(train_dataset)
        if iteration == 1:
            train_plot = vis.line(X=np.array([iteration]),
                                  Y=np.array([cost]),
                                  opts=dict(
                                      title='Train Cost layer 1',
                                      xlabel='Iteration',
                                      ylabel='BCELoss'
                                  ))
        else:
            vis.updateTrace(X=np.array([iteration]),
                            Y=np.array([cost]),
                            win=train_plot)

    ### layer 2
    print("=========== Layer 2 ===========")
    parameter = chain(vae.encodeLinear2.parameters(), vae.decodeLinear2.parameters())
    optim = torch.optim.SGD(parameter, lr=learning_rate, momentum=momentum)
    iteration = 0
    for epoch in range(1, epochs + 1):
        print("Epoch: {}".format(epoch))
        vae.train()
        cost = 0.0
        iteration += 1
        for x, _ in train_loader:
            # original = Variable(x)
            if torch.cuda.is_available():
                x = x.cuda()
            X = Variable(corruptInput(x, corruption_level))
            h = vae.encode_layer(X, 1)
            original = h
            vae.zero_grad()

            h = vae.forward_layer(h, 2)
            # h = vae.decode_layer(h, 1).view(original.size())
            reconstruction = h
            loss = BceCriterion(reconstruction, original)
            if sparsity:
                loss += SparsityCost(vae.activation)

            loss.backward()
            optim.step()
            vae.clear_activation()
            cost += loss.data[0]*len(x)
        cost /= len(train_dataset)
        if iteration == 1:
            train_plot = vis.line(X=np.array([iteration]),
                                  Y=np.array([cost]),
                                  opts=dict(
                                      title='Train Cost layer 2',
                                      xlabel='Iteration',
                                      ylabel='BCELoss'
                                  ))
        else:
            vis.updateTrace(X=np.array([iteration]),
                            Y=np.array([cost]),
                            win=train_plot)

    ### layer 3
    print("=========== Layer 3 ===========")
    parameter = chain(vae.encodeLinear3.parameters(), vae.decodeLinear3.parameters())
    optim = torch.optim.SGD(parameter, lr=learning_rate, momentum=momentum)
    iteration = 0
    for epoch in range(1, epochs + 1):
        print("Epoch: {}".format(epoch))
        vae.train()
        cost = 0.0
        iteration += 1
        for x, _ in train_loader:
            # original = Variable(x)
            if torch.cuda.is_available():
                x = x.cuda()
            X = Variable(corruptInput(x, corruption_level))
            h = vae.encode_layer(X, 1)
            h = vae.encode_layer(h, 2)
            original = h
            vae.zero_grad()

            h = vae.forward_layer(h, 3)
            # h = vae.decode_layer(h, 2)
            # h = vae.decode_layer(h, 1).view(original.size())
            reconstruction = h
            loss = BceCriterion(reconstruction, original)
            if sparsity:
                loss += SparsityCost(vae.activation)

            loss.backward()
            optim.step()
            vae.clear_activation()
            cost += loss.data[0] * len(x)
        cost /= len(train_dataset)
        if iteration == 1:
            train_plot = vis.line(X=np.array([iteration]),
                                  Y=np.array([cost]),
                                  opts=dict(
                                      title='Train Cost layer 3',
                                      xlabel='Iteration',
                                      ylabel='BCELoss'
                                  ))
        else:
            vis.updateTrace(X=np.array([iteration]),
                            Y=np.array([cost]),
                            win=train_plot)

    ### Visualize Weights
    vae.apply(visualizeLinearLayerWeight)

    ### Visualize Activation
    vae.eval()
    test_loader = data.DataLoader(test_dataset, shuffle=True, batch_size=100)
    for x, _ in test_loader:
        X = Variable(x)
        if torch.cuda.is_available():
            X = X.cuda()
        reconstruction = vae(X)
        # reconstructed images
        images = torch.cat([X.data, reconstruction.data], 3).cpu().numpy()
        vis.images(images, nrow=5, opts=dict(title='original-reconstruction'))
        X = X.view(X.size()[0], -1)
        # encode 1 (784 -> 900)
        X = vae.encode_layer(X, 1)
        image_x = X.view(-1, 1, 30, 30)
        vis.images(image_x.cpu().data.numpy(), nrow=5, opts=dict(title='Sigmoid-EncodeLinear-1'))
        # encode 2 (900 -> 625)
        X = vae.encode_layer(X, 2)
        image_x = X.view(-1, 1, 25, 25)
        vis.images(image_x.cpu().data.numpy(), nrow=5, opts=dict(title='Sigmoid-EncodeLinear-2'))
        # encode 3 (625 -> 400)
        X = vae.encode_layer(X, 3)
        # image_x = normalizeForImage(X)[0].view(-1, 1, 20, 20)
        # vis.images(image_x.cpu().data.numpy(), nrow=5, opts=dict(title='Sigmoid-EncodeLinear-3'))
        # decode 3 (400 -> 625)
        X = vae.decode_layer(X, 3)
        image_x = X.view(-1, 1, 25, 25)
        vis.images(image_x.cpu().data.numpy(), nrow=5, opts=dict(title='Sigmoid-DecodeLinear-3'))
        # decode 2 (625 -> 900)
        X = vae.decode_layer(X, 2)
        image_x = X.view(-1, 1, 30, 30)
        vis.images(image_x.cpu().data.numpy(), nrow=5, opts=dict(title='Sigmoid-DecodeLinear-2'))
        break

    if classify:
        # parameter = chain(vae.encodeLinear1.parameters(),
        #                   vae.encodeLinear2.parameters(),
        #                   vae.encodeLinear3.parameters())
        optim = torch.optim.SGD(vae.parameters(), lr=learning_rate, momentum=momentum)
        classifier_loss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            classifier_loss = classifier_loss.cuda()
        iteration = 0
        for epoch in range(1,epochs+1):
            vae.train()
            iteration += 1
            cost = 0
            for x, y in train_loader:
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                X, Y = Variable(x), Variable(y)
                vae.zero_grad()

                pred = vae.classify(X)
                loss = classifier_loss(pred, Y)

                loss.backward()
                optim.step()
                cost += loss.data[0] * len(x)
            cost /= len(train_dataset)

            vae.eval()
            accuracy = 0
            for x_test, y_test in test_loader:
                if torch.cuda.is_available():
                    x_test, y_test = x_test.cuda(), y_test.cuda()
                X_test = Variable(x_test)
                _, pred = torch.max(vae.classify(X_test).data, 1)
                accuracy += (pred == y_test).sum()
            accuracy = 100 * accuracy / len(test_dataset)

            #plot loss and accuracy here
            if iteration == 1:
                softmax_train_plot = vis.line(X=np.array([iteration]),
                                              Y=np.array([cost]),
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
                                Y=np.array([cost]),
                                win=softmax_train_plot)
                vis.updateTrace(X=np.array([iteration]),
                                Y=np.array([accuracy]),
                                win=softmax_test_plot)

if __name__ == '__main__':
    main(classify=False, sparsity=False)
