from model import model
from reader import Reader
import tensorflow as tf
import numpy as np
import time

import sys


# import matplotlib.pyplot as plt
# import numpy as np

FILE_NAME = './dio.txt'

r = Reader.Reader()
r.read(FILE_NAME,encode_words=False)

batch_size = 32
n_classes = r.get_unique_words()
n_timesteps = 256
n_target = 256
n_input = 256
n_rnn = 256
n_layers = 1

learning_rate = 0.001

# if(len(sys.argv) == 2):
#     epochs = sys.argv[1]
x = tf.placeholder(tf.float32, [batch_size, n_timesteps, n_classes])
y = tf.placeholder(tf.float32,[batch_size, n_timesteps, n_classes])

sess = tf.InteractiveSession()

r.init_batch(n_timesteps, n_timesteps,step=n_timesteps)

pred, cost, train_step, accuracy = model.RNN(x, y, n_timesteps, n_rnn, n_classes, batch_size)

sess.run(tf.global_variables_initializer())

# saver = tf.train.Saver()

losses = []
accuracies = []

total_loss = 0
total_acc = 0

n_batches = len(r.data)//batch_size


print(n_batches)
start_time = time.clock()

epochs = 1000

print('-------------')
print("Starting, {} epochs.".format(epochs))
print("Data size {}".format(len(r.data)))
print("N classes {}".format(n_classes))
print("N batches {}".format(n_batches))

for n in range(epochs):

    # for i in range(n_batches):
        X = []
        Y = []
        for _ in range(batch_size):
            inputs, targets = r.next()

            X.append([r.encode(input) for input in inputs])

            Y.append([r.encode(target) for target in targets])

            # TODO we can actually just build the X and shift it
            #  by one to create the targets ;)
        # X = np.array(X)
        # Y = np.array(Y)
        # print(X.shape, Y.shape)
        x_hot = tf.one_hot(X, depth=n_classes, on_value=1.0)
        y_hot = tf.one_hot(Y, depth=n_classes, on_value=1.0)
        # print(n)
        output, loss, acc = sess.run([ train_step, cost, accuracy ], feed_dict={x: x_hot.eval(), y: y_hot.eval()})

        # losses.append(loss)
        # accuracies.append(acc)
        #
        total_loss += loss
        total_acc += acc

        if(n % 10 == 0 and n > 0):
            print('--------')
            print('Iterations: {}'.format(n))
            print('AVG Loss: {0:.4f}'.format(total_loss/(10)))
            print('AVG Acc: {0:.4f}'.format(total_acc/(10)))
            total_loss = 0
            total_acc = 0
            # save_path = saver.save(sess, "/tmp/model-{}.ckpt".format(time.strftime("%H:%M:%S")))
            # print("Model saved in file: %s" % save_path)
        # if(n % n_batches == 0):
            # read all the dataset
            # print('---------------')
            # print('Iterations {}'.format(n))
            # r.init_batch(n_timesteps,n_timesteps,step=n_timesteps)

# save_path = saver.save(sess, "/tmp/model.ckpt")
# print("Model saved in file: %s" % save_path)

N_TEXT = 100

finish_time = time.clock()

total_time = finish_time - start_time

print('----------')
print('Finish After: {0:.4f}s'.format(total_time))
print('Epoches per second: {0:.4f}'.format(total_time/n))

exit(1)
# plt.plot(losses, label="losses")
# plt.plot(accuracies, label="accuracies")
# plt.legend()
# plt.show()
