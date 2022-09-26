import tensorflow as tf
from tensorflow import keras


class HanAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HanAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        bulid
        """
        self.dense_tanh = keras.layers.Dense(units=input_shape[-1], activation="tanh")
        self.dense = keras.layers.Dense(units=1, activation=None)
        self.softmax = keras.layers.Softmax()

    def call(self, inputs, training=None, **kwargs):
        """
        call
        """
        out_tanh = self.dense_tanh(inputs)
        softmax_tanh = self.softmax(tf.squeeze(self.dense(out_tanh), axis=[-1]))
        outputs = tf.matmul(tf.transpose(inputs, perm=[0, 2, 1]), tf.expand_dims(softmax_tanh, -1))
        outputs = tf.squeeze(outputs, axis=[-1])
        return outputs

    def compute_output_shape(self, input_shape):
        """
        :param input_shape:
        :return shape
        """
        output_shape = (input_shape[0], input_shape[-1])
        return output_shape

    def get_config(self):
        """
        get_config
        """
        base_config = super(HanAttention, self).get_config()
        return base_config
