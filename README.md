# cz4042
NTU CZ4042: Neural Network

---

## Project 1:

### A: Landsat Satellite dirt MLP classifier

- dataset: https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)

### B:

- dataset:

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
3. FullyConnected(180, 100) > ReLU
4. FullyConnected(100, 10)

**Criterion**

CrossEntropyLoss ~ LogSoftmax > NegativeLogLikelihoodLoss

**Optimizer**

epochs = 40
1. SGD(batch_size=128, lr=0.5, decay=10<sup>-6</sup>)
2. SGD(batch_size=128, lr=0.5, decay=10<sup>-6</sup>, momentum=0.5)
3. RMSProp(batch_size=128, lr=10<sup>-3</sup>, decay=10<sup>-4</sup>, alpha=0.9, epsilon=10<sup>-6</sup>)

**Note**
- All visualization produced using [Visdom](https://github.com/facebookresearch/visdom)
- Code using [PyTorch](http://pytorch.org/)
- Run visdom then check and comment/uncomment part of the code in main() that you want to run(visdom is assumed to run on default port)

### B: MNIST VAE
