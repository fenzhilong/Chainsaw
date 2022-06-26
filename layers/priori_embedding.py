import tensorflow as tf
from tensorflow import keras
import pickle


class PrioriEmbedding(keras.layers.Layer):
    def __init__(self, dim, embedding_file, **kwargs):
        super(PrioriEmbedding, self).__init__(**kwargs)
        self.dim = dim
        # embedding_file e.x is 'data/self_embedding.pkl'
        self.embeddings_file = embedding_file

    def build(self, input_shape):
        """
        bulid
        """
        self.kernel = self.add_weight(name='weight', shape=(2, self.dim), initializer='glorot_uniform')
        with open(self.embeddings_file, 'rb') as fr:
            self.embeddings = pickle.load(fr)
        self.embeddings = tf.convert_to_tensor(self.embeddings, dtype=tf.float32)
        self.embeddings = tf.concat((self.kernel, self.embeddings), axis=0)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        """
        call
        """
        inputs = tf.cast(inputs, dtype=tf.int32)
        out = tf.nn.embedding_lookup(self.embeddings, inputs)
        return out

    def compute_output_shape(self, input_shape):
        """
        :param input_shape:
        :return shape
        """
        output_shape = (input_shape[0],input_shape[1], self.dim)
        return output_shape

    def get_config(self):
        """
        get_config
        """
        base_config = super(PrioriEmbedding, self).get_config()
        base_config['dim'] = self.dim
        return base_config
