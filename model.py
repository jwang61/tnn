import tensorflow as tf
import numpy as np

# --- RNN PARAMETERS ---
# The embedding dimension
EMBEDDING_DIM = 256

# Number of LSTM units
UNITS = [512, 512]
GRAD_CLIP = 5

class Model():
    def __init__(self, vocab_size, batch_size_, max_tweet_len_, training=True):
        #self.units = UNITS
        batch_size = batch_size_ if training else 1
        seq_len = max_tweet_len_ - 1 if training else 1

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_len], name="Placeholder")
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_len], name="Target")

        rnn_layers = [tf.contrib.rnn.LSTMBlockCell(size) for size in UNITS]
        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)

        self.zero_state = rnn_cell.zero_state(batch_size, tf.float32)

        # Convert each character id into a vector representation
        embedding_matrix = tf.get_variable('embedding_matrix', [vocab_size, EMBEDDING_DIM],
                                           initializer=tf.random_uniform_initializer(minval=-1.,
                                                                                     maxval=1.))
        embedding = tf.nn.embedding_lookup(embedding_matrix, self.input_data)

        rnn_out, state = tf.nn.dynamic_rnn(rnn_cell, embedding, initial_state=self.zero_state,
                                           dtype=tf.float32)

        logits = tf.layers.dense(rnn_out, vocab_size, name='dense')
        self.prediction = tf.nn.softmax(logits)

        self.last_state = state

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets,
                                                           logits=logits,
                                                           name="loss"))

        training_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, training_vars), GRAD_CLIP)
        self.lr = tf.Variable(0.0, trainable=False)

        tf.summary.scalar('loss', self.loss)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, training_vars))

    def sample(self, sess, char2idx, idx2char):
        # You can change the start string to experiment
        start_string = '<'

        char = start_string
        ret_str = start_string
        for _ in range(MAX_TWEET_LEN):
            x = np.zeros((1, 1))
            x[0, 0] = char2idx[char]

            prediction = sess.run(self.prediction, {self.input_data : x})
            p = prediction[0]
            idx = np.argmax(p)

            next_ = idx2char[idx]
            ret_str += next_
            char = next_

        return ret_str
