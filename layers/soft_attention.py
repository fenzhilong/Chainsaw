import tensorflow as tf
from tensorflow import keras


class SoftAttentionMask(keras.layers.Layer):
    def __init__(self):
        super(SoftAttentionMask, self).__init__()

    def build(self, input_shape):
        self.layer_softmax = keras.layers.Softmax(axis=-1)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        mask = inputs[1]

        matrix = tf.matmul(x[0], tf.transpose(x[1], perm=[0, 2, 1]))
        attention = self.layer_softmax(matrix, tf.transpose(mask[1], perm=[0, 2, 1]))

        matrix_T = tf.transpose(matrix, perm=[0, 2, 1])
        attention_T = self.layer_softmax(matrix_T, tf.transpose(mask[0], perm=[0, 2, 1]))

        A = tf.matmul(attention, x[1])
        B = tf.matmul(attention_T, x[0])

        return [A*mask[0], B*mask[1]]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class SoftAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SoftAttention, self).__init__()
        self.supports_masking = True

    def call(self, inputs, mask=None, **kwargs):
        matrix = tf.matmul(inputs[0], tf.transpose(inputs[1], perm=[0, 2, 1]))
        attention = tf.nn.softmax(matrix, axis=2)

        matrix_T = tf.transpose(matrix, perm=[0, 2, 1])
        attention_T = tf.nn.softmax(matrix_T, axis=2)

        A = tf.matmul(attention, inputs[1])
        B = tf.matmul(attention_T, inputs[0])

        return A, B
