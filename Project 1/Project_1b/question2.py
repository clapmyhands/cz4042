import matplotlib.pyplot as plt
import numpy as np

import function as f

np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden1 = 30  # num of neurons in hidden layer 1
folds = 5
alpha_values = [0.001, 0.5 * 0.001, 0.0001, 0.5 * 0.0001, 0.00001]


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

len_trainX = len(trainX)
interval = len_trainX // 5
best_alpha = -1
fold_cost = []
best_average = 10000000000
best_train = None
best_test = None
best_valuesDict = {}
train = None
test = None
valuesDict = {}
plt.figure()

for alpha in alpha_values:
    print('For Alpha = %.5f' % alpha)
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
        best_alpha = alpha
        best_train = train
        best_test = test
        best_valuesDict = valuesDict

    plt.plot(range(epochs), valuesDict['test_cost'], label='Alpha = %.5f' % alpha)

print('Best model found at Alpha  = %.5f' % best_alpha)

# Plots
# a) Plot the training error against number of epochs for the 3-layer network.
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Validation Errors at Various Alphas')
plt.legend()
plt.savefig('partb_2a.png')
plt.show()

train, test, valuesDict = f.function(trainX, trainY, testX, testY, no_hidden1,
                                             best_alpha, epochs, batch_size)


# b) Plot the final test errors of prediction by the network.
plt.figure()
plt.plot(range(epochs), valuesDict['test_cost'], label='test error')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Test Errors at Alpha = %.5f' % best_alpha)
plt.legend()
plt.savefig('partb_2b.png')
plt.show()