import tensorflow as tf
from tensorflow import keras
from layers.han_attention import HanAttention
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class HAN(keras.Model):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            num_classes,
            sentence_length,
            doc_length,
            hidden_size,
            dropout_rate=None
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.doc_length = doc_length
        self.num_classes = num_classes
        self.sentence_length = sentence_length
        self.doc_length = doc_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        if dropout_rate is None:
            self.dropout_rate = 0.3
        super(HAN, self).__init__()

        self.embedding = keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.bilstm_word = keras.layers.Bidirectional(
            keras.layers.LSTM(self.hidden_size, return_sequences=True), merge_mode='concat')
        self.attention_word = HanAttention()

        self.vector_sentence = keras.layers.TimeDistributed(self.word_encoder())

        self.bilstm_sentence = keras.layers.Bidirectional(
            keras.layers.LSTM(self.hidden_size, return_sequences=True), merge_mode='concat')
        self.attention_sentence = HanAttention()

        self.dense = keras.layers.Dense(units=self.num_classes, activation=tf.nn.softmax)
        self.dropout = keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None)

    def word_encoder(self):
        inputs = keras.layers.Input(shape=(self.sentence_length, ))
        embedding = self.embedding(inputs)
        output_word = self.bilstm_word(embedding)
        output_word = self.attention_word(output_word)
        return tf.keras.Model(inputs=inputs, outputs=output_word)

    def call(self, inputs, training=None, mask=None):
        outputs = self.vector_sentence(inputs)
        outputs = self.bilstm_sentence(outputs)
        outputs = self.attention_sentence(outputs)
        outputs = self.dropout(outputs)
        outputs = self.dense(outputs)
        return outputs

    def model(self):
        text = tf.keras.Input(shape=(self.doc_length, self.sentence_length))
        return tf.keras.Model(inputs=text, outputs=self.call(text))
