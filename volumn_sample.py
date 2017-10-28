import os
import tensorflow as tf
import numpy as np

def _repeat(x, num_repeat):
    rep  = tf.ones((1, num_repeat), tf.int32)
    x = tf.reshape(x, [-1, 1])
    x = tf.reshape(tf.matmul(x, rep), [-1])
    return x

def _meshgrid(height, width):
    '''return x_s: [height, width], y_s: [height, width]
    '''
    x = tf.expand_dims(tf.linspace(0.0, 1.0, width), dim=0)
    y = tf.expand_dims(tf.linspace(0.0, 1.0, height), dim=1)
    x_s = tf.matmul(tf.ones((height, 1)), x)
    y_s = tf.matmul(y, tf.ones((1, width)))
    return x_s, y_s

def _sample(X, x, y):
    '''X: input img [batch_size, h, w, c]
     x: sample x coordinate [batch_size*h*w]
     y: sample y coordinate [batch_size*h*w]
     t: weight of the image [batch_size*h*w]
     return: [batch_size*h*w, 3]
    '''
    [batch_size, h, w, channel] = X.get_shape().as_list()
    w_f = tf.cast(w, 'float32')
    h_f = tf.cast(h, 'float32')

    x = x * w_f
    y = y * h_f

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, w - 1)
    x1 = tf.clip_by_value(x1, 0, w - 1)
    y0 = tf.clip_by_value(y0, 0, h - 1)
    y1 = tf.clip_by_value(y1, 0, h - 1)

    dim2 = w
    dim1 = h * w
    base = _repeat(tf.range(batch_size) * dim1, h * w)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y0 + x1
    idx_c = base_y1 + x0
    idx_d = base_y1 + x1

    im_flat = tf.reshape(X, [-1, channel])
    Ia = tf.gather(im_flat, idx_a)  # [batch_size*ht*wt, 3]
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    wa = tf.expand_dims((x1_f - x) * (y1_f - y), axis=1)
    wb = tf.expand_dims((x - x0_f) * (y1_f - y), axis=1)
    wc = tf.expand_dims((x1_f - x) * (y - y0_f), axis=1)
    wd = tf.expand_dims((x - x0_f) * (y - y0_f), axis=1)

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out

def volum_sample(X, F):
    ''' X: input video [batch_size, frame, h, w, c]
     F: 3D voxel flow field: [batch_size, h, w, 3]
     return: output image [batch_size, h, w, c]
     '''

    [batch_size, frame, h, w, chanel] = X.get_shape().as_list()
    x_s, y_s = _meshgrid(h, w)
    x_s = tf.reshape(x_s, [-1])
    y_s = tf.reshape(y_s, [-1])
    x_s = tf.tile(x_s, tf.stack([batch_size])) #[batch_size*h*w]
    y_s = tf.tile(y_s, tf.stack([batch_size]))
    delta_x = tf.reshape(F[...,0], [-1])
    delta_y = tf.reshape(F[...,1], [-1])

    # t-1 frame
    x0 = x_s - delta_x
    y0 = y_s - delta_y
    out0 = _sample(X[:,0], x0, y0)
    x1 = x_s + delta_x
    y1 = y_s + delta_y
    out1 = _sample(X[:,1], x1, y1)
    delta_t = tf.reshape(F[..., -1], [-1])
    delta_t = tf.expand_dims(delta_t, axis=1)
    out = out0 * (1 - delta_t) + out1 * delta_t
    out = tf.reshape(out, [batch_size, h, w, chanel])
    return out










