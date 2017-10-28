import tensorflow as tf
import numpy as np
import cv2
from volumn_sample import *
import os

batch_size = 2
seqlen = 2
h = 400
w = 400
c = 3

def imshow(out):
    out = np.array(out * 255, dtype=np.uint8)
    cv2.imshow('img3', out[0])
    cv2.waitKey()
    cv2.imshow('img4', out[1])
    cv2.waitKey()

def get_batch():
    DIMS = (400, 400)
    CAT1 = 'cat1.jpg'
    CAT2 = 'cat2.jpg'

    img1 = cv2.imread(CAT1)
    img2 = cv2.imread(CAT2)

    img1 = np.array(cv2.resize(img1, DIMS)) / 255.0
    img2 = np.array(cv2.resize(img2, DIMS)) / 255.0

    t_img1 = np.array(img1 * 255.0, dtype=np.uint8)
    t_img2 = np.array(img2 * 255.0, dtype=np.uint8)
    cv2.imshow('img1', t_img1)
    cv2.waitKey()
    cv2.imshow('img2', t_img2)
    cv2.waitKey()

    img1 = np.reshape(img1, (1, 1, 400, 400, 3))
    img1 = np.concatenate((img1, img1), axis=1)
    img2 = np.reshape(img2, (1, 1, 400, 400, 3))
    img2 = np.concatenate((img2, img2), axis=1)
    img = np.concatenate((img1, img2), axis=0)
    return img

def get_F():
    f = np.ones((2, 400, 400, 3))
    f = f / 2.0
    return f

X = tf.placeholder('float32', [batch_size, seqlen, h, w, c])
F = tf.placeholder('float32', [batch_size, h, w, 3])
out = volum_sample(X, F)

sess = tf.Session()
x0 = get_batch()
f = get_F()
out_img = sess.run(out, feed_dict={X: x0, F: f})
imshow(out_img)

