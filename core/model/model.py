import tensorflow as tf
from tensorflow.contrib import rnn


def RNN(x,y,n_input, n_hidden, n_output,n_layer = 1):

    w = tf.Variable(tf.random_normal([n_hidden, n_output]))
    b = tf.Variable(tf.random_normal([n_output]))

    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_input, 1)

    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    # initial_state = rnn_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)

    # rnn.DropoutWrapper(rnn_cell,output_keep_prob=0.5)
    # rnn_cell = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden),0.5), rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden//2),0.5)])
    #
    # rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for _ in range(4)])

    outputs, state = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)

    pred = tf.matmul(outputs[-1], w) + b

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    train_step = tf.train.RMSPropOptimizer(0.001).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # there are n_input outputs but
    # we only want the last output
    return pred, cost, train_step, accuracy


