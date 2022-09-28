import tensorflow as tf
import keras
from keras import layers
from keras.regularizers import l2


class CNNBlock(keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super(CNNBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        """
        bulid
        """
        self.conv1 = keras.layers.Conv1D(
            self.num_filters, kernel_size=self.kernel_size, padding="same",
            kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))
        self.conv2 = keras.layers.Conv1D(
            self.num_filters, kernel_size=self.kernel_size, padding="same",
            kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.prelu1 = keras.layers.PReLU()
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.prelu2 = keras.layers.PReLU()

    def call(self, inputs, training=None, **kwargs):
        """
        call
        """
        outputs = self.conv1(inputs)
        outputs = self.batch_norm1(outputs)
        outputs = self.prelu1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.batch_norm2(outputs)
        outputs = self.prelu2(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        """
        :param input_shape:
        :return shape
        """
        output_shape = input_shape
        return output_shape

    def get_config(self):
        """
        get_config
        """
        base_config = super(CNNBlock, self).get_config()
        base_config["num_filters"] = self.num_filters
        base_config["kernel_size"] = self.kernel_size
        return base_config
