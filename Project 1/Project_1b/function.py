import time

import numpy as np
import theano
import theano.tensor as T

np.random.seed(10)

hidden = 3
no_hidden1 = 30
floatX = theano.config.floatX


def init_bias(n = 1):
    return theano.shared(np.zeros(n), theano.config.floatX)


def init_weights(n_in=1, n_out=1, logistic=True):
    np.random.seed(10)
    W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out))
    if logistic == True:
        W_values *= 4
    return theano.shared(W_values, theano.config.floatX)


def set_bias(b, n = 1):
    b.set_value(np.zeros(n))


def set_weights(w, n_in=1, n_out=1, logistic=True):
    np.random.seed
    W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out))
    if logistic == True:
        W_values *= 4
    w.set_value(W_values)


# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)


def normalize(X, X_mean, X_std):
    return (X - X_mean) / X_std


def shuffle_data(samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    # print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


def scale_normalize(data):
    # scale and normalize data
    data_max, data_min = np.max(data, axis=0), np.min(data, axis=0)
    data = scale(data, data_min, data_max)
    data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
    data = normalize(data, data_mean, data_std)

    return data


def generateFunction(learning_rate, no_hidden=hidden):
    x = T.matrix('x')  # data sample
    d = T.matrix('d')  # desired output

    w_o = init_weights(no_hidden, 1)
    b_o = init_bias()
    w_h1 = init_weights(1, no_hidden)
    b_h1 = init_bias(no_hidden)

    alpha = theano.shared(learning_rate, floatX)

    # Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
    y = T.dot(h1_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d - y)))
    accuracy = T.mean(d - y)

    # define gradients
    dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

    train = theano.function(
        inputs=[x, d],
        outputs=cost,
        updates=[[w_o, w_o - alpha * dw_o],
                 [b_o, b_o - alpha * db_o],
                 [w_h1, w_h1 - alpha * dw_h],
                 [b_h1, b_h1 - alpha * db_h]],
        allow_input_downcast=True
    )

    test = theano.function(
        inputs=[x, d],
        outputs=[y, cost, accuracy],
        allow_input_downcast=True
    )

    parameter_dict = {
        'w_o': w_o,
        'b_o': b_o,
        'w_h1': w_h1,
        'b_h1': b_h1
    }

    return train, test, parameter_dict


def runTrain(train, test, trainX, trainY, testX, testY, epochs, batch_size, parameter_dict):
    train_cost = np.zeros(epochs)
    test_cost = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    no_features = trainX.shape[1]

    min_error = 1e+15
    best_iter = 0
    best_w_o = np.zeros(no_hidden1)
    best_w_h1 = np.zeros([no_features, no_hidden1])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden1)
    values_dict = {}

    for iter in range(epochs):
        if iter % 100 == 0:
            print(iter)

        trainX, trainY = shuffle_data(trainX, trainY)

        costs = []

        for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
            # print "start: %d, end: %d \n" % (start, end)
            indiv_train_cost = train(trainX[start:end:], np.transpose(trainY[start:end]))
            costs.append(indiv_train_cost)

        train_cost[iter] = np.mean(costs)
        pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

        if test_cost[iter] < min_error:
            best_iter = iter
            min_error = test_cost[iter]
            best_w_o = w_o.get_value()
            best_w_h1 = w_h1.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()

    # set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)

    best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

    values_dict.update({'best_iter': best_iter, 'best_w_o': best_w_o, 'best_w_h1': best_w_h1,
                        'best_b_o': best_b_o, 'best_b_h1': best_b_h1, 'best_pred': best_pred,
                        'best_cost': best_cost, 'best_accuracy': best_accuracy,
                        'train_cost': train_cost, 'test_cost': test_cost})

    print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d' % (best_cost, best_accuracy, best_iter))


def function(trainX, trainY, testX, testY, no_hidden1, learning_rate, epochs, batch_size):
    no_features = trainX.shape[1]
    x = T.matrix('x')  # data sample
    d = T.matrix('d')  # desired output
    no_samples = T.scalar('no_samples')

    t = time.time()

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(no_hidden1) * .01, floatX)
    b_o = theano.shared(np.random.randn() * .01, floatX)
    w_h1 = theano.shared(np.random.randn(no_features, no_hidden1) * .01, floatX)
    b_h1 = theano.shared(np.random.randn(no_hidden1) * 0.01, floatX)

    # learning rate
    alpha = theano.shared(learning_rate, floatX)

    # Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
    y = T.dot(h1_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d - y)))
    accuracy = T.mean(d - y)

    # define gradients
    dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

    train = theano.function(
        inputs=[x, d],
        outputs=cost,
        updates=[[w_o, w_o - alpha * dw_o],
                 [b_o, b_o - alpha * db_o],
                 [w_h1, w_h1 - alpha * dw_h],
                 [b_h1, b_h1 - alpha * db_h]],
        allow_input_downcast=True
    )

    test = theano.function(
        inputs=[x, d],
        outputs=[y, cost, accuracy],
        allow_input_downcast=True
    )

    train_cost = np.zeros(epochs)
    test_cost = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    min_error = 1e+15
    best_iter = 0
    best_w_o = np.zeros(no_hidden1)
    best_w_h1 = np.zeros([no_features, no_hidden1])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden1)
    values_dict = {}

    alpha.set_value(learning_rate)

    for iter in range(epochs):
        if iter % 100 == 0:
            print(iter)

        trainX, trainY = shuffle_data(trainX, trainY)

        costs = []

        for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
            # print "start: %d, end: %d \n" % (start, end)
            indiv_train_cost = train(trainX[start:end:], np.transpose(trainY[start:end]))
            costs.append(indiv_train_cost)

        train_cost[iter] = np.mean(costs)
        pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

        if test_cost[iter] < min_error:
            best_iter = iter
            min_error = test_cost[iter]
            best_w_o = w_o.get_value()
            best_w_h1 = w_h1.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()

    # set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)

    best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

    values_dict.update({'best_iter': best_iter, 'best_w_o': best_w_o, 'best_w_h1': best_w_h1,
                        'best_b_o': best_b_o, 'best_b_h1': best_b_h1, 'best_pred': best_pred,
                        'best_cost': best_cost, 'best_accuracy': best_accuracy,
                        'train_cost': train_cost, 'test_cost': test_cost})

    print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d' % (best_cost, best_accuracy, best_iter))

    return train, test, values_dict