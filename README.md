# cz4042
NTU CZ4042: Neural Network

---

## Project 1:

### A: Landsat Satellite dirt MLP classifier

- dataset: https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)

### B: California Housing

- dataset: http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

---

## Project 2:

- dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)

### A: MNIST CNN Classifier

**Data**

- Training set: 60000 samples
- Test set: 10000 samples

**Model Architecture**

1. Convolutional(kernel_size=[9,9], stride=1, padding=0) > MaxPool(2) > ReLU 
2. Convolutional(kernel_size=[5,5], stride=1, padding=0) > MaxPool(2) > ReLU 
3. Linear(180, 100) > ReLU
4. Linear(100, 10)

**Criterion**

- CrossEntropyLoss ~ LogSoftmax > NegativeLogLikelihoodLoss

**Optimizer & Parameters**

epochs = 40
1. SGD(batch_size=128, lr=0.5, decay=10<sup>-6</sup>)
2. SGD(batch_size=128, lr=0.5, decay=10<sup>-6</sup>, momentum=0.5)
3. RMSProp(batch_size=128, lr=10<sup>-3</sup>, decay=10<sup>-4</sup>, alpha=0.9, epsilon=10<sup>-6</sup>)


### B: MNIST VAE
**Data**
- Training set: 12000 samples

**Model Architecture**

Layer is trained 1-by-1. Train layer 1. Use output of first encoding as input to train Layer 2. Use output of second encoding to train Layer 3.
<br>Visualization of weights are normalized to [0,1]

Layer 1:
1. Linear(784, 900) > Sigmoid
2. Linear(900, 784) > Sigmoid

Layer 2:
1. Linear(900, 625) > Sigmoid
2. Linear(625, 900) > Sigmoid

Layer 3:
1. Linear(625, 400) > Sigmoid
2. Linear(400, 625) > Sigmoid

**Criterion**
- Autoencoder:= BinaryCrossEntropyLoss
- Classifier:= CrossEntropyLoss
- Sparsity constraint:= KL-Divergence

**Optimizer & Parameters**

Epochs = 30
Corruption = 0.1
1. SGD(batch_size=128, lr=0.1) ~ w/o sparsity constraint
2. SGD(batch_size=128, lr=0.1, momentum=0.1) w/ sparsity constraint


**Note**
- All visualization produced using [Visdom](https://github.com/facebookresearch/visdom) - assumed to run on default port
- Code using [PyTorch](http://pytorch.org/)
- Change the flag in main() to change training and or
- comment/uncomment part of the code
