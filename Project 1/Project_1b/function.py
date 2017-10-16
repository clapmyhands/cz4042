import time

import numpy as np
import theano
import theano.tensor as T

np.random.seed(10)

hidden = 3
floatX = theano.config.floatX


def init_bias(n = 1):
    #return theano.shared(np.zeros(n) if n != 1 else 0, floatX)
    return theano.shared(np.zeros(n), floatX)


def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out))
    if logistic == True:
        W_values *= 4
    return theano.shared(W_values, floatX)

# scale and normalize input data
def scale(data):
    data_max, data_min = np.max(data, axis=0), np.min(data, axis=0)
    return (data - data_min) / (data_max - data_min)


def normalize(data):
    data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
    return (data - data_mean) / data_std


def shuffle_data(samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    # print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


def scale_normalize(data):
    # scale and normalize data
    data_scale = scale(data)
    data_normalize = normalize(data_scale)

    return data_normalize


def function_nlayer(no_layers, trainX, trainY, testX, testY, hidden_neurons, learning_rate, epochs, batch_size):
    if no_layers == 3:
        return function(trainX, trainY, testX, testY, hidden_neurons[3], learning_rate, epochs, batch_size)
    elif no_layers == 4:
        return function_4layer(trainX, trainY, testX, testY, hidden_neurons[3], hidden_neurons[4], learning_rate, epochs, batch_size)
    elif no_layers == 5:
        return function_5layer(trainX, trainY, testX, testY, hidden_neurons[3], hidden_neurons[4], hidden_neurons[5],
                        learning_rate, epochs, batch_size)


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
            indiv_train_cost = train(trainX[start:end], np.transpose(trainY[start:end]))
            costs.append(indiv_train_cost)

        if batch_size == len(trainX):
            train_cost[iter] = train(trainX, np.transpose(trainY))
        else:
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

    print('Minimum error: %.6f, Best accuracy %.6f, Number of Iterations: %d' % (best_cost, best_accuracy, best_iter))

    return train, test, values_dict


def function_4layer(trainX, trainY, testX, testY, no_hidden1, no_hidden2, learning_rate, epochs, batch_size):
    no_features = trainX.shape[1]
    x = T.matrix('x')  # data sample
    d = T.matrix('d')  # desired output
    no_samples = T.scalar('no_samples')

    t = time.time()

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(no_hidden2) * .01, floatX)
    b_o = theano.shared(np.random.randn() * .01, floatX)
    w_h1 = theano.shared(np.random.randn(no_features, no_hidden1) * .01, floatX)
    b_h1 = theano.shared(np.random.randn(no_hidden1) * 0.01, floatX)
    w_h2 = theano.shared(np.random.randn(no_hidden1, no_hidden2) * .01, floatX)
    b_h2 = theano.shared(np.random.randn(no_hidden2) * 0.01, floatX)

    # learning rate
    alpha = theano.shared(learning_rate, floatX)

    # Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
    h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
    y = T.dot(h2_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d - y)))
    accuracy = T.mean(d - y)

    # define gradients
    dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2 = T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2])

    train = theano.function(
        inputs=[x, d],
        outputs=cost,
        updates=[[w_o, w_o - alpha * dw_o],
                 [b_o, b_o - alpha * db_o],
                 [w_h1, w_h1 - alpha * dw_h1],
                 [b_h1, b_h1 - alpha * db_h1],
                 [w_h2, w_h2 - alpha * dw_h2],
                 [b_h2, b_h2 - alpha * db_h2]],
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
    best_w_h2 = np.zeros([no_features, no_hidden2])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden1)
    best_b_h2 = np.zeros(no_hidden2)
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
            best_w_h2 = w_h2.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()
            best_b_h2 = b_h2.get_value()

    # set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)
    w_h2.set_value(best_w_h2)
    b_h2.set_value(best_b_h2)

    best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

    values_dict.update({'best_iter': best_iter, 'best_w_o': best_w_o, 'best_w_h1': best_w_h1,
                        'best_w_h2': best_w_h2, 'best_b_o': best_b_o, 'best_b_h1': best_b_h1,
                        'best_b_h2': best_b_h2, 'best_pred': best_pred, 'best_cost': best_cost,
                        'best_accuracy': best_accuracy, 'train_cost': train_cost, 'test_cost': test_cost})

    print('Minimum error: %.6f, Best accuracy %.6f, Number of Iterations: %d' % (best_cost, best_accuracy, best_iter))

    return train, test, values_dict


def function_5layer(trainX, trainY, testX, testY, no_hidden1, no_hidden2, no_hidden3, learning_rate, epochs, batch_size):
    no_features = trainX.shape[1]
    x = T.matrix('x')  # data sample
    d = T.matrix('d')  # desired output
    no_samples = T.scalar('no_samples')

    t = time.time()

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(no_hidden2) * .01, floatX)
    b_o = theano.shared(np.random.randn() * .01, floatX)
    w_h1 = theano.shared(np.random.randn(no_features, no_hidden1) * .01, floatX)
    b_h1 = theano.shared(np.random.randn(no_hidden1) * 0.01, floatX)
    w_h2 = theano.shared(np.random.randn(no_hidden1, no_hidden2) * .01, floatX)
    b_h2 = theano.shared(np.random.randn(no_hidden2) * 0.01, floatX)
    w_h3 = theano.shared(np.random.randn(no_hidden2, no_hidden3) * .01, floatX)
    b_h3 = theano.shared(np.random.randn(no_hidden3) * 0.01, floatX)

    # learning rate
    alpha = theano.shared(learning_rate, floatX)

    # Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
    h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
    h3_out = T.nnet.sigmoid(T.dot(h2_out, w_h3) + b_h3)
    y = T.dot(h3_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d - y)))
    accuracy = T.mean(d - y)

    # define gradients
    dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2, dw_h3, db_h3 = T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2, w_h3, b_h3])

    train = theano.function(
        inputs=[x, d],
        outputs=cost,
        updates=[[w_o, w_o - alpha * dw_o],
                 [b_o, b_o - alpha * db_o],
                 [w_h1, w_h1 - alpha * dw_h1],
                 [b_h1, b_h1 - alpha * db_h1],
                 [w_h2, w_h2 - alpha * dw_h2],
                 [b_h2, b_h2 - alpha * db_h2],
                 [w_h3, w_h3 - alpha * dw_h3],
                 [b_h3, b_h3 - alpha * db_h3]],
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
    best_w_h2 = np.zeros([no_features, no_hidden2])
    best_w_h3 = np.zeros([no_features, no_hidden3])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden1)
    best_b_h2 = np.zeros(no_hidden2)
    best_b_h3 = np.zeros(no_hidden3)
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
            best_w_h2 = w_h2.get_value()
            best_w_h3 = w_h3.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()
            best_b_h2 = b_h2.get_value()
            best_b_h3 = b_h3.get_value()

    # set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)
    w_h2.set_value(best_w_h2)
    b_h2.set_value(best_b_h2)
    w_h3.set_value(best_w_h3)
    b_h3.set_value(best_b_h3)

    best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

    values_dict.update({'best_iter': best_iter, 'best_w_o': best_w_o, 'best_w_h1': best_w_h1,
                        'best_w_h2': best_w_h2, 'best_w_h3': best_w_h3,'best_b_o': best_b_o,
                        'best_b_h1': best_b_h1, 'best_b_h2': best_b_h2, 'best_b_h3': best_b_h3,
                        'best_pred': best_pred, 'best_cost': best_cost, 'best_accuracy': best_accuracy,
                        'train_cost': train_cost, 'test_cost': test_cost})

    print('Minimum error: %.6f, Best accuracy %.6f, Number of Iterations: %d' % (best_cost, best_accuracy, best_iter))

    return train, test, values_dict
