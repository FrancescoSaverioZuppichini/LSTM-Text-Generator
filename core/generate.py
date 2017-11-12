import tensorflow as tf
from model import model

def generate_from(checkpoint_name, start_text, n_text):
    print(checkpoint_name, start_text, n_text)
    tf.reset_default_graph()

    n_classes = 255

    my_model = model.RNN([512,512,512], n_classes, 0.001)

    my_model.build()

    with tf.Session() as sess:

            saver = tf.train.Saver()

            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_name))

            text = my_model.generate(start_text, sess, interactive=True, n_text=n_text)

            return text

