import tensorflow as tf
from parser import args
from model import model

def generate_from(checkpoint_name, start_text, n_text):

    tf.reset_default_graph()

    n_classes = 255

    my_model = model.RNN.from_args(args, n_classes)

    my_model.build()

    with tf.Session() as sess:

            saver = tf.train.Saver()

            saver.restore(sess, tf.train.latest_checkpoint("./checkpoints/{}".format(checkpoint_name)))

            text = my_model.generate(start_text, sess, interactive=True, n_text=n_text)

            return text

