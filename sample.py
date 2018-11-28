import pickle
import tensorflow as tf
import numpy as np

MODEL_FILE = "newmodel.pb"
VOCAB_FILE = "vocab.pkl"

START_CHAR = "<"

with open(VOCAB_FILE, 'rb') as f:
    vocab = sorted(pickle.load(f))

# Creating a mapping from unique characters to indices
char2idx = {u : i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
vocab_size = len(vocab)

graph_def = tf.GraphDef()
with tf.gfile.Open(MODEL_FILE, 'rb') as f:
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    # Placeholder Substitution
    tf.import_graph_def(graph_def, name="")

    with tf.Session() as sess:
        #TODO: run session
        print("done")
    
