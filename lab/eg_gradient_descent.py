import tensorflow as tf
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
import keras.optimizers as opt


opt = tf.keras.optimizers.SGD(learning_rate=0.1)
var1 = tf.Variable([10.0, 20.0])
var2 = tf.Variable(100.0)
loss = lambda: (var1 ** 2 + var2 ** 2)/2.0       # d(loss)/d(var1) == var1
for i in range(1000):
    step_count = opt.minimize(loss, [var1, var2]).numpy()
    # The first step is `-learning_rate*sign(grad)`
    print(var1.numpy(), var2.numpy())
# w = tf.Variable(tf.constant(5, dtype=tf.float32))
print("women")


w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.3
epoch = 50

t1 = time.time()
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)

    w.assign_sub(lr * grads)
    print(epoch, w.numpy(), loss)
print(time.time() - t1)
