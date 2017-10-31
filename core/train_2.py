from model import model
from reader import Reader
import tensorflow as tf
import numpy as np
import time

import sys


# import matplotlib.pyplot as plt
# import numpy as np

FILE_NAME = './dio.txt'

CHECK_POINT = False



# variables
batch_size = 64
# sequence len
n_timesteps = 256
n_target = 256
n_input = n_target
# size of each rnn
n_rnn = [256, 128]
n_layers = 1

learning_rate = 0.001

r = Reader.Reader(batch_size=batch_size, sequence_len=n_timesteps)
r.read(FILE_NAME,encode_words=False)
n_classes = r.get_unique_words()

x = tf.placeholder(tf.float32, [batch_size, n_timesteps, n_classes])
y = tf.placeholder(tf.float32,[batch_size, n_timesteps, n_classes])

sess = tf.InteractiveSession()

pred, cost, train_step, accuracy, correct_pred = model.RNN(x, y, n_timesteps, n_rnn, n_classes, batch_size)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

losses = []
accuracies = []

total_loss = 0
total_acc = 0

n_batches = len(r.data)//(batch_size * n_timesteps)

epochs = 30
start_time = time.clock()

print('-------------')
print("Starting, {} epochs.".format(epochs))
print("Data size {}".format(len(r.data)))
print("N classes {}".format(n_classes))
print("N batches {}".format(n_batches))
print('Start at {}'.format(start_time))

X,Y = r.create_training_set()

for n in range(epochs):

        for i in range(len(X)//batch_size):
            x_batch,y_batch = r.next(i)

            x_hot = tf.one_hot(x_batch, depth=n_classes, on_value=1.0)
            y_hot = tf.one_hot(y_batch, depth=n_classes, on_value=1.0)

            _, loss, acc = sess.run([ train_step, cost, accuracy ], feed_dict={x: x_hot.eval(), y: y_hot.eval()})

            total_loss += loss
            total_acc += acc

        print('--------')
        print('Iterations: {}'.format(n))
        print('AVG Loss: {0:.4f}'.format(total_loss/(len(X)//batch_size)))
        print('AVG Acc: {0:.4f}'.format(total_acc/(len(X)//batch_size)))
        # print('Iter per second: {0:.4f}'.format(time.clock() - start_time_inner / 10/1000))
        # start_time_inner = time.clock()

        total_loss = 0
        total_acc = 0

        if(CHECK_POINT):
            save_path = saver.save(sess, "/tmp/model-{}.ckpt".format(time.strftime("%H:%M:%S")))
            print("Model saved in file: %s" % save_path)
#
# x_hot = tf.one_hot(X, depth=n_classes, on_value=1.0)
#
# ps = sess.run(pred,feed_dict={x: x_hot.eval()})
# print(int(tf.argmax(pred, 1).eval()))
# print(ps)
# print(''.join([chr(p) for p in ps]))

save_path = saver.save(sess, "/tmp/model.ckpt")
print("Model saved in file: %s" % save_path)

N_TEXT = 100

finish_time = time.clock()

total_time = finish_time - start_time

print('----------')
print('Finish After: {0:.4f}s'.format(total_time))
print('Iterations per second: {0:.4f}'.format(total_time/epochs))

exit(1)
