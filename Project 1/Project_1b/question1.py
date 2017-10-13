import matplotlib.pyplot as plt
import numpy as np

import function as f

np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden1 = 30  # num of neurons in hidden layer 1
learning_rate = 0.0001
folds = 5


# read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
Y_data = (np.asmatrix(Y_data)).transpose()

X_data, Y_data = f.shuffle_data(X_data, Y_data)

# separate train and test data
m = 3 * X_data.shape[0] // 10
testX, testY = X_data[:m], Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

trainX = f.scale_normalize(trainX)
testX = f.scale_normalize(testX)

train, test, valuesDict = f.function(trainX, trainY, testX, testY, no_hidden1,
                                     learning_rate, epochs, batch_size)

train_cost = valuesDict['train_cost']
test_cost = valuesDict['test_cost']

# Plots
# a) Plot the training error against number of epochs for the 3-layer network.
plt.figure()
plt.plot(range(epochs), train_cost, label='train error')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Training Errors at Alpha = %.4f' % learning_rate)
plt.legend()
plt.savefig('partb_1a.png')
plt.show()

# b) Plot the final test errors of prediction by the network.
plt.figure()
plt.plot(range(epochs), test_cost, label='test error')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Test Errors at Alpha = %.4f' % learning_rate)
plt.legend()
plt.savefig('partb_1b.png')
plt.show()