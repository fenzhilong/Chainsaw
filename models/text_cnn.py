import tensorflow as tf
from tensorflow import keras
from layers.priori_embedding import PrioriEmbedding
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TextCnn(keras.Model):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            num_filters,
            num_classes,
            sequence_length,
            kernel_sizes=None,
            dropout_rate=None
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.kernel_sizes = kernel_sizes
        if kernel_sizes is None:
            self.kernel_sizes = [2, 3, 4]
        if dropout_rate is None:
            self.dropout_rate = 0.3
        super(TextCnn, self).__init__()

        # self.embedding = PrioriEmbedding(self.embedding_dim, "data/self_embedding.pkl", name='embedding')
        self.embedding = keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.convs = [
            keras.layers.Conv1D(self.num_filters, kernel_size=kernel_size, activation='relu')
            for kernel_size in self.kernel_sizes]
        self.poolings = [
            keras.layers.MaxPool1D(pool_size=self.sequence_length+1-kernel_size, padding='valid')
            for kernel_size in self.kernel_sizes]

        self.flatten = keras.layers.Flatten(name='flatten')
        self.dense = keras.layers.Dense(units=self.num_classes, activation=tf.nn.softmax)
        self.dropout = keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None)

    def call(self, inputs, training=None, mask=None):
        embedding = self.embedding(inputs)
        outputs = []
        for i in range(len(self.kernel_sizes)):
            conv = self.convs[i](embedding)
            pooling = self.poolings[i](conv)
            outputs.append(pooling)

        output = tf.concat(outputs, axis=1)
        output = self.flatten(output)

        output = self.dropout(output)
        output = self.dense(output)
        return output

    def model(self):
        text = tf.keras.Input(shape=(self.sequence_length, ), dtype='int64')
        return tf.keras.Model(inputs=text, outputs=self.call(text))
