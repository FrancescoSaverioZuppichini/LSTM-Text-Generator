import tensorflow as tf

from model import model


def generate_from(checkpoint_name, start_text, n_text, layers, eta):

    print(checkpoint_name)

    tf.reset_default_graph()

    n_classes = 255

    my_model = model.RNN(layers, n_classes, eta)

    my_model.build()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()

    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_name))

    text = my_model.generate(start_text, sess, interactive=False, n_text=n_text)

    return text

