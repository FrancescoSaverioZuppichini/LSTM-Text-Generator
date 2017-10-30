from model import model
from reader import Reader
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = './divina_commedia.txt'

r = Reader.Reader()
r.read(FILE_NAME,encode_words=False)

batch_size = 32
n_classes = r.get_unique_words()
n_timesteps = 3
n_target = 3
n_input = 1

n_rnn = 512
n_layers = 1

learning_rate = 0.001
epochs = 10000
sess = tf.InteractiveSession()

print(n_classes)
x = tf.placeholder(tf.float32, [batch_size, n_timesteps, n_classes])
y = tf.placeholder(tf.float32,[batch_size, n_timesteps, n_classes])

r.init_batch(n_timesteps, n_timesteps)

pred, cost, train_step, accuracy = model.RNN(x, y, n_timesteps, n_rnn, n_classes, batch_size)

sess.run(tf.global_variables_initializer())

losses = []
accuracies = []

total_loss = 0
total_acc = 0

n_batches = len(r.data)//batch_size

for n in range(epochs):
    X = []
    Y = []

    for _ in range(batch_size):
        inputs, targets = r.next()

        X.append([r.encode(input) for input in inputs])

        Y.append([r.encode(target) for target in targets])

    # TODO we can actually just build the X and shift it
    #  by one to create the targets ;)
    # print(X, "----", Y)
    x_hot = tf.one_hot(X, depth=n_classes, on_value=1.0)
    y_hot = tf.one_hot(Y, depth=n_classes, on_value=1.0)
    # print(x_hot)
    # print(y_hot)
    # print(X.shape, Y.shape)
    # Y = np.reshape(Y, [batch_size, n_classes])
    # print(X)
    output, loss, acc = sess.run([ train_step, cost, accuracy ], feed_dict={x: x_hot.eval(), y: y_hot.eval()})
    #
    losses.append(loss)
    accuracies.append(acc)
    total_loss += loss
    total_acc += acc
    #
    #
    if(n % 100 == 0):
    #
        print('--------')
        print('Iterations: {}'.format(n))
        print('AVG Loss: {}'.format(total_loss/(100)))
        print('AVG Acc: {}'.format(total_acc/(100)))
        total_loss = 0
        total_acc = 0

plt.plot(losses, label="losses")
plt.plot(accuracies, label="accuracies")
plt.legend()
plt.show()
