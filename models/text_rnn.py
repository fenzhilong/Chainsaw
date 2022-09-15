import tensorflow as tf
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TextRNN(keras.Model):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            num_classes,
            sequence_length,
            dropout_rate=None
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate

        if dropout_rate is None:
            self.dropout_rate = 0.3
        super(TextRNN, self).__init__()

        # self.embedding = PrioriEmbedding(self.embedding_dim, "data/self_embedding.pkl", name='embedding')
        self.embedding = keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.bilstm = keras.layers.Bidirectional(
            keras.layers.LSTM(self.hidden_size, return_sequences=True), merge_mode='concat')
        self.dropout = keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None)
        self.lstm = keras.layers.LSTM(self.hidden_size)
        self.dense = keras.layers.Dense(units=self.num_classes, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        embedding = self.embedding(inputs)
        output = self.bilstm(embedding)
        output = self.lstm(output)
        output = self.dropout(output)
        output = self.dense(output)
        return output

    def model(self):
        text = tf.keras.Input(shape=(self.sequence_length, ), dtype='int64')
        return tf.keras.Model(inputs=text, outputs=self.call(text))
