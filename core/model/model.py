import tensorflow as tf
from tensorflow.contrib import rnn


def RNN(x, y, n_input, n_hidden, n_classes, n_batch, n_layer = 1):

    w = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    b = tf.Variable(tf.random_normal([n_classes]))

    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    initial_state = rnn_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=initial_state ,dtype=tf.float32)

    # rnn.DropoutWrapper(rnn_cell,output_keep_prob=0.5)
    # rnn_cell = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden),0.5), rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden//2),0.5)])
    #
    # rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for _ in range(4)])
    # pack
    pack_pred = tf.reshape(outputs, [-1 , n_hidden])

    pred = tf.matmul(pack_pred, w) + b

    flat_Y = tf.reshape(y, [-1 , n_classes])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=flat_Y))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(flat_Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # there are n_input outputs but
    # we only want the last output
    return pred, cost, train_step, accuracy


