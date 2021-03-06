""" Sampling script to generate tweets """
import pickle
import tensorflow as tf
import numpy as np
from model import Model

# Location of model checkpoint
SAVE_DIR = "save"
VOCAB_FILE = "vocab.pkl"

START_CHAR = "<"

def sample():
    # Load vocabulary file
    with open(VOCAB_FILE, 'rb') as f:
        vocab = sorted(pickle.load(f))

    char2idx = {u : i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    vocab_size = len(vocab)

    # Create model
    model = Model(vocab_size, 1, 1)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)

        if ckpt and ckpt.model_checkpoint_path:
            # Run sampling
            saver.restore(sess, ckpt.model_checkpoint_path)
            tweet = model.sample(sess, char2idx, idx2char)
            print(tweet)

if __name__ == "__main__":
    sample()
