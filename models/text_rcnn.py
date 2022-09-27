import tensorflow as tf
import keras
from keras import layers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TextRCNN(keras.Model):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            rnn_hidden_size,
            num_filters,
            num_classes,
            sequence_length,
            kernel_size=1,
            dropout_rate=None
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        if dropout_rate is None:
            self.dropout_rate = 0.3

        super(TextRCNN, self).__init__()

        # self.embedding = PrioriEmbedding(self.embedding_dim, "data/self_embedding.pkl", name='embedding')
        self.embedding = keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.forward_lstm = keras.layers.LSTM(self.rnn_hidden_size, return_sequences=True)
        self.backward_lstm = keras.layers.LSTM(self.rnn_hidden_size, return_sequences=True, go_backwards=True)
        self.conv = keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, activation='relu')
        self.dropout = keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None)
        self.pooling = keras.layers.GlobalMaxPooling1D()
        self.dense = keras.layers.Dense(units=self.num_classes, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        embedding = self.embedding(inputs)
        forward_output = self.forward_lstm(embedding)
        backward_output = tf.reverse(self.backward_lstm(embedding), axis=[1])
        output = tf.concat([forward_output, embedding, backward_output],  axis=2)

        output = self.conv(output)
        output = self.dropout(output)
        output = self.pooling(output)
        output = self.dense(output)
        return output

    def model(self):
        text = tf.keras.Input(shape=(self.sequence_length, ), dtype='int64')
        return tf.keras.Model(inputs=text, outputs=self.call(text))
