import tensorflow as tf
import keras
from keras import layers
from layers.cnn_block import CNNBlock
from keras.regularizers import l2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DPCNN(keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_filters,
        num_classes,
        sentence_length,
        hidden_size=100,
        num_layers=8,
        kernel_size=3,
        max_pool_size=3,
        max_pool_strides=2,
        dropout_rate=None
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.sentence_length = sentence_length
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.max_pool_size = max_pool_size
        self.max_pool_strides = max_pool_strides
        self.dropout_rate = dropout_rate
        if dropout_rate is None:
            self.dropout_rate = 0.3
        super(DPCNN, self).__init__()

        self.embedding = keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_dropout = keras.layers.SpatialDropout1D(self.dropout_rate)
        self.conv1 = keras.layers.Conv1D(
            self.num_filters, kernel_size=1, padding="same", kernel_regularizer=l2(),
            bias_regularizer=l2(), activation="linear")

        self.cnn_blocks = [CNNBlock(num_filters=self.num_filters, kernel_size=self.kernel_size)
                           for _ in range(num_layers)]

        self.global_pooling = keras.layers.GlobalMaxPooling1D()
        self.dense1 = keras.layers.Dense(units=self.hidden_size, activation="linear")
        self.batch_norm = keras.layers.BatchNormalization()
        self.prelu1 = keras.layers.PReLU()
        self.prelu2 = keras.layers.PReLU()
        self.dropout = keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None)
        self.dense2 = keras.layers.Dense(units=self.num_classes, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        embedding = self.embedding(inputs)
        region_embedding = self.conv1(embedding)
        region_embedding = self.embedding_dropout(region_embedding)
        region_embedding = self.prelu1(region_embedding)

        inputs = region_embedding
        output_add = None
        for cnn_block in self.cnn_blocks:
            output_block = cnn_block(inputs)
            output_add = keras.layers.Add()([inputs, output_block])
            inputs = keras.layers.MaxPooling1D(pool_size=self.max_pool_size, strides=self.max_pool_strides)(output_add)
            if inputs.shape[-2] <= 2:
                break

        outputs = self.global_pooling(output_add)
        outputs = self.dense1(outputs)
        outputs = self.batch_norm(outputs)
        outputs = self.prelu2(outputs)
        outputs = self.dropout(outputs)
        outputs = self.dense2(outputs)
        return outputs

    def model(self):
        text = tf.keras.Input(shape=(self.sentence_length,))
        return tf.keras.Model(inputs=text, outputs=self.call(text))
