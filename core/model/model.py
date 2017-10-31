import tensorflow as tf
from tensorflow.contrib import rnn


def RNN(x, y, n_input, n_hidden, n_classes, n_batch, n_layer = 1):

    w = tf.Variable(tf.random_normal([n_hidden[-1], n_classes]))
    b = tf.Variable(tf.random_normal([n_classes]))

    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n) for n in n_hidden])
    # rnn_cell = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden),0.5), rnn.BasicLSTMCell(n_hidden)])

    initial_state = rnn_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)

    outputs, state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=initial_state ,dtype=tf.float32)

    pack_pred = tf.reshape(outputs, [-1 , n_hidden[-1]])

    pred = tf.matmul(pack_pred, w) + b

    flat_Y = tf.reshape(y, [-1 , n_classes])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=flat_Y))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(flat_Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return pred, cost, train_step, accuracy, correct_pred


