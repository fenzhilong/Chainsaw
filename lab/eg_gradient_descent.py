import tensorflow as tf
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
#
