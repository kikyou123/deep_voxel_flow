import os
import tensorflow as tf
from op import  *

def gen_flow(X, dim = 16):
    '''
    :param X:  input video of size [batch_size, seqlen, h, w, c]
    :return: voxel flow F [batch_size, h, w, 3]
    '''
    [batch_size, seqlen, h, w, c] = X.get_shape().as_list()
    X = tf.transpose(X, (0, 2, 3, 1, 4))
    X = tf.reshape(X, [batch_size, h, w, seqlen * c])

    encoder1 = conv2d(X, dim, k_h=1, k_w=1, d_h=1, d_w=1, name='encoder1')
    encoder2 = conv2d(relu(encoder1), dim * 2, k_h=5, k_w=5, name='encoder2')
    encoder3 = conv2d(relu(encoder2), dim * 4, k_h=5, k_w=5, name='encoder3')
    encoder4 = conv2d(relu(encoder3), dim * 4, k_h=3, k_w=3, name='encoder4')
    decoder4 = deconv2d(relu(encoder4), dim * 4, k_h=3, k_w=3, d_h=1, d_w=1, name='decoder4')
    decoder4 = tf.concat(values=[encoder4, decoder4], axis=-1)
    decoder3 = deconv2d(relu(decoder4), dim * 4, k_h=3, k_w=3, name='decoder3')
    decoder3 = tf.concat(values=[encoder3, decoder3], axis=-1)
    decoder2 = deconv2d(relu(decoder3), dim * 2, k_h=5, k_w=5, name='decoder2')
    decoder2 = tf.concat(values=[encoder2, decoder2], axis=-1)
    decoder1 = deconv2d(relu(decoder2), dim, k_h=5, k_w=5, name='decoder1')
    decoder1 = relu(decoder1)
    F = tf.tanh(conv2d(decoder1, 3, k_h=1, k_w=1, d_h=1, d_w=1, name='f'))
    F= tf.concat(values=(F[...,0:1], F[...,1:2], (F[..., 2:] + 1.0) / 2.0),axis=-1)
    return F

def get_loss(target, gen_img):
    """
    :param target: label of size [batch_size, h, w, 3]
    :param gen_img: generate image of size [batch_size, h, w, 3]
    :return: loss
    """
    loss = tf.reduce_mean(tf.abs(target - gen_img))
    return loss

def create_optimizers(loss, params, learning_rate):
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = params)
    return opt


