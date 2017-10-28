import tensorflow as tf
import numpy as np

tmp = tf.linspace(-1.0, 1.0, 2)
sess = tf.Session()
print sess.run(tmp)


x0 = np.array([[[2, 4],[1, 3]],[[1, 5],[1, 0]]])
f = np.array([[[0.5, 0.5], [0, 0]], [[0.5, 0], [0.5, 0]], [[0.5, 0.5], [0.5, 0.5]]])
f = np.transpose(f, (1, 2, 0))
x0 = np.reshape(x0, [batch_size, seqlen, h, w, c])
f = np.reshape(f, [batch_size, h, w, 3])
X = tf.convert_to_tensor(x0, dtype=tf.float32)
F = tf.convert_to_tensor(f, dtype=tf.float32)
sess = tf.Session()