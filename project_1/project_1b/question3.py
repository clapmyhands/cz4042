import matplotlib.pyplot as plt
import numpy as np

import function as f

np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden1_values = [20, 30, 40, 50, 60]  # num of neurons in hidden layer 1
folds = 5
alpha = 0.001


# read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
Y_data = (np.asmatrix(Y_data)).transpose()
Y_data = Y_data/1e6

# Scale X_data then shuffle data
X_data = f.scale(X_data)
X_data, Y_data = f.shuffle_data(X_data, Y_data)

# separate train and test data
m = 3 * X_data.shape[0] // 10
testX, testY = X_data[:m], Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

len_trainX = len(trainX)
interval = len_trainX // 5
best_no_hidden1 = -1
fold_cost = []
best_average = 1e+15
best_train = None
best_test = None
best_valuesDict = {}
train = None
test = None
valuesDict = {}

for no_hidden1 in no_hidden1_values:
    print('For hidden layer neurons = %d' % no_hidden1)
    for fold in range(folds):
        start, end = fold * interval, (fold + 1) * interval
        if (end + interval) > len_trainX:
            end = len_trainX

        validateX_kfold, validateY_kfold = trainX[start:end], trainY[start:end]
        trainX_kfold, trainY_kfold = np.append(trainX[:start], trainX[end:], axis=0), np.append(trainY[:start], trainY[end:], axis=0)

        train, test, valuesDict = f.function(trainX_kfold, trainY_kfold, validateX_kfold, validateY_kfold, no_hidden1,
                                             alpha, epochs, batch_size)

        fold_cost = np.append(fold_cost, np.mean(valuesDict['test_cost']))

    average_fold_cost = np.mean(fold_cost)
    if average_fold_cost < best_average:
        best_average = average_fold_cost
        best_no_hidden1 = no_hidden1
        best_train = train
        best_test = test
        best_valuesDict = valuesDict

    plt.figure(0)
    plt.plot(range(epochs), valuesDict['train_cost'], label='Hidden layer neurons = %d' % no_hidden1)
    plt.figure(1)
    plt.plot(range(epochs), valuesDict['test_cost'], label='Hidden layer neurons = %d' % no_hidden1)

print('Best model found at hidden layer neurons  = %d' % best_no_hidden1)

# Plots
# a) Plot the training error against number of epochs for the 3-layer network.
plt.figure(0)
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Training Errors at Various # of Hidden Layer Neurons')
plt.legend()
plt.savefig('partb_3a_train.png')

plt.figure(1)
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Validation Errors at Various # of Hidden Layer Neurons')
plt.legend()
plt.savefig('partb_3a_validate.png')

train, test, valuesDict = f.function(trainX, trainY, testX, testY, best_no_hidden1, alpha, epochs, batch_size)


# b) Plot the final test errors of prediction by the network.
plt.figure(2)
plt.plot(range(epochs), valuesDict['test_cost'], label='test error')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Test Errors at # Hidden Layer Neurons = %d' % best_no_hidden1)
plt.legend()
plt.savefig('partb_3b_test.png')
plt.show()