"""
Script to train model
"""
import time
import pickle
import random
import os

import tensorflow as tf
import numpy as np
from model import Model

# TODO: Add Argparse
TRAIN = True

DATA_FILE = "tweets.txt"
VOCAB_FILE = "vocab.pkl"
SAVE_DIR = "save"

SPLIT_CHAR = ">"

LOG_DIR = "logs"

# --- Training Params ---
EPOCHS = 20
LEARNING_RATE = 0.001
DECAY_RATE = 0.92
BATCH_SIZE = 32
MAX_TWEET_LEN = 313

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def train():
    with open(DATA_FILE, 'r') as f:
        text = f.read()
    with open(VOCAB_FILE, 'rb') as f:
        vocab = sorted(pickle.load(f))

    char2idx = {u : i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    vocab_size = len(vocab)

    data = []
    index = 0
    for i, char in enumerate(text):
        if char == SPLIT_CHAR:
            tweet = text[index:i+1]
            while len(tweet) < MAX_TWEET_LEN:
                tweet += ">"
            text_as_int = np.array([char2idx[c] for c in tweet])
            data.append(split_input_target(text_as_int))
            index = i+1

    num_tweets = len(data)
    batch_count = num_tweets//BATCH_SIZE

    model = Model(vocab_size, BATCH_SIZE, MAX_TWEET_LEN-1)

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    ckpt_path = os.path.join(SAVE_DIR, "model.ckpt")
    with tf.Session() as sess:
        summary_ops = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(LOG_DIR)
        summary_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        
        # TODO: Add restoring ckpt
        for epoch in range(EPOCHS):
            random.shuffle(data)
            start = time.time()
            lr = LEARNING_RATE * (DECAY_RATE ** epoch)
            sess.run(tf.assign(model.lr, lr))

            for i in range(batch_count):
                state = sess.run(model.zero_state)
                batch_input = []
                batch_target = []
                for j in range(BATCH_SIZE):
                    batch_input.append(data[i*BATCH_SIZE + j][0])
                    batch_target.append(data[i*BATCH_SIZE + j][1])
                input_ = np.array(batch_input)
                target_ = np.array(batch_target)
                feed_dict = {model.input_data : input_, model.targets : target_,
                             model.zero_state : state}
                summary, state, loss, _ = sess.run([summary_ops, model.last_state,
                                                    model.loss, model.train_op], feed_dict)

                summary_writer.add_summary(summary, epoch*batch_count + i)

                if i % 100 == 0:
                    saver.save(sess, ckpt_path, epoch*batch_count + i)
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, i, loss))
                    

            print('Time taken for 1 epoch {} sec, lr : {}\n'.format(time.time() - start, lr))
        saver.save(sess, ckpt_path, EPOCHS*batch_count)
        #tf.train.write_graph(sess.graph, "./", "model2.pb", as_text=False)

if __name__ == "__main__":
    train()
