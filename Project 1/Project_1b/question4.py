import matplotlib.pyplot as plt
import numpy as np

import function as f

np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden1 = 60  # num of neurons in hidden layer 1
no_hidden2 = 20
no_hidden3 = 20
folds = 5
alpha = 0.0001


# read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
Y_data = (np.asmatrix(Y_data)).transpose()
Y_data = Y_data/1e3

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
hidden_layer_dict = {
    3: no_hidden1,
    4: no_hidden2,
    5: no_hidden3
}

for hidden_layer in [3, 4, 5]:
    print('For # of hidden layers = %d' % hidden_layer)
    for fold in range(folds):
        start, end = fold * interval, (fold + 1) * interval
        if (end + interval) > len_trainX:
            end = len_trainX

        validateX_kfold, validateY_kfold = trainX[start:end], trainY[start:end]
        trainX_kfold, trainY_kfold = np.append(trainX[:start], trainX[end:], axis=0), np.append(trainY[:start], trainY[end:], axis=0)

        train, test, valuesDict = f.function_nlayer(hidden_layer, trainX_kfold, trainY_kfold, validateX_kfold,
                                                    validateY_kfold, hidden_layer_dict, alpha, epochs, batch_size)

        fold_cost = np.append(fold_cost, np.mean(valuesDict['test_cost']))

    average_fold_cost = np.mean(fold_cost)
    if average_fold_cost < best_average:
        best_average = average_fold_cost
        best_hidden_layer = hidden_layer
        best_train = train
        best_test = test
        best_valuesDict = valuesDict

    plt.figure(0)
    plt.plot(range(epochs), valuesDict['train_cost'], label='Number of layers = %d' % hidden_layer)
    plt.figure(1)
    plt.plot(range(epochs), valuesDict['test_cost'], label='Number of layers = %d' % hidden_layer)

print('Best model found at number of layers  = %d' % best_hidden_layer)

# Plots
# a) Plot the training error against number of epochs for the n-layer network.
plt.figure(0)
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Training Errors at Various # of Layers')
plt.legend()
plt.savefig('partb_4a_train.png')
plt.figure(1)
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Validation Errors at Various # of Layers')
plt.legend()
plt.savefig('partb_4a_validate.png')

train, test, valuesDict = f.function_nlayer(hidden_layer, trainX, trainY, testX, testY, hidden_layer_dict, alpha, epochs, batch_size)


# b) Plot the final test errors of prediction by the network.
plt.figure(2)
plt.plot(range(epochs), valuesDict['test_cost'], label='test error')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Error')
plt.title('Test Errors at # of Layers = %d' % best_hidden_layer)
plt.legend()
plt.savefig('partb_4b_test.png')
plt.show()