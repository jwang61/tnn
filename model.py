"""
Contains the class for the LSTM model
"""
import tensorflow as tf
import numpy as np

# --- RNN PARAMETERS ---
# The embedding dimension
EMBEDDING_DIM = 256

# Number of LSTM units
UNITS = [512, 512]

# Gradient Clipping
GRAD_CLIP = 5

def softmax(x):
    """(Found on stack overflow) Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Model():
    """ LSTM Model with sampling function"""
    def __init__(self, vocab_size, batch_size_, max_tweet_len_):
        batch_size = batch_size_
        seq_len = max_tweet_len_

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_len], name="Placeholder")
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_len], name="Target")

        rnn_layers = [tf.contrib.rnn.LSTMBlockCell(size) for size in UNITS]
        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)

        self.zero_state = rnn_cell.zero_state(batch_size, tf.float32)

        embedding_matrix = tf.get_variable('embedding_matrix', [vocab_size, EMBEDDING_DIM],
                                           initializer=tf.random_uniform_initializer(minval=-1.,
                                                                                     maxval=1.))
        embedding = tf.nn.embedding_lookup(embedding_matrix, self.input_data)

        rnn_out, state = tf.nn.dynamic_rnn(rnn_cell, embedding, initial_state=self.zero_state,
                                           dtype=tf.float32)

        self.logits = tf.layers.dense(rnn_out, vocab_size, name='dense')

        self.last_state = state

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets,
                                                           logits=self.logits,
                                                           name="loss"))

        training_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, training_vars), GRAD_CLIP)
        self.lr = tf.Variable(0.0, trainable=False)

        tf.summary.scalar('loss', self.loss)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, training_vars))

    def sample(self, sess, char2idx, idx2char, start_string="", temp=1, max_len=300):
        """ 
        Sampling function that return generated tweet.
        Use start_string to change initial characters
        Use temp < 1 for wilder guesses, temp > 1 for more rigid guesses
        """
        start = '<' + start_string
        state = sess.run(self.zero_state)

        # Setting up the initial state of the model based on start_string
        for char in start[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = char2idx[char]

            prediction, state = sess.run([self.logits, self.last_state],
                                         {self.input_data : x, self.zero_state : state})

        char = start[-1]
        ret_str = start[1:]

        for _ in range(max_len):
            x = np.zeros((1, 1))
            x[0, 0] = char2idx[char]

            prediction, state = sess.run([self.logits, self.last_state],
                                         {self.input_data : x, self.zero_state : state})
            p = softmax(np.squeeze(prediction) * temp)

            cumsum = np.cumsum(p)
            rand = np.random.rand(1) * np.sum(p)
            idx = np.searchsorted(cumsum, rand[0])

            next_ = idx2char[idx]
            if next_ == ">":
                break
            ret_str += next_
            char = next_

        return ret_str
