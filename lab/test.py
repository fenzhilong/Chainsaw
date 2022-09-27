import tensorflow as tf
import keras
from keras import layers

dense = keras.layers.Dense(units=1, activation=None)
softmax = keras.layers.Softmax()

inputs = tf.random.normal([23, 30, 200])
print(inputs)

softmax_tanh = softmax(tf.squeeze(dense(inputs), axis=[-1]))
print(softmax_tanh)
print(tf.reverse(inputs, axis=[1]))
lstm = tf.keras.layers.LSTM(4)
output = lstm(inputs)
print(output.shape)
print(output)

lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
print(whole_seq_output)
print(whole_seq_output.shape)

print(final_memory_state.shape)
print(final_memory_state)

print(final_carry_state.shape)

lstm = tf.keras.layers.LSTM(4, return_state=True)
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
print(whole_seq_output.shape)
print(whole_seq_output.shape)

print(final_memory_state.shape)
print(final_memory_state)

print(final_carry_state.shape)


