import tensorflow as tf
import os
import numpy as np
import importlib
import random
from model import *
from dataset_mnist import *
from volumn_sample import *
from util import  *


batch_size = 16
seqlen = 2
h = 64
w = 64
c = 1
lr = 0.0001
random_seed = 0
max_iter = 10000

out = open('log.txt', 'w')
checkpoint_dir = 'checkpoint'
train_dir = 'train'
test_dir = 'test'
gt_dir = 'gt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(gt_dir):
    os.makedirs(gt_dir)


def setup_tensorflow():
    config = tf.ConfigProto()
    sess = tf.Session(config = config)
    with sess.graph.as_default():
        tf.set_random_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    return sess

def next_batch(dh):
    track = dh.GetBatch()
    track = np.transpose(track, [0, 2, 3, 1, 4]) # [batch_size, height, width, channel, seqlen]
    x0 = np.expand_dims(track[..., 0], axis=1)
    x1 = np.expand_dims(track[..., 2], axis=1)
    batch_input = np.concatenate((x0, x1), axis=1)
    batch_target = track[..., 1]
    return batch_input, batch_target

def _train():
    X = tf.placeholder('float32', [batch_size, seqlen, h, w, c])
    Y = tf.placeholder('float32', [batch_size, h, w, c])
    F = gen_flow(X)
    gen_img = volum_sample(X, F)
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    loss = get_loss(Y, gen_img)
    opt = create_optimizers(loss, vars, lr)

    sess = setup_tensorflow()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print 'train from %s' % (ckpt.model_checkpoint_path)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print "train from scratch..."

    print('prepare data...')
    dataset = importlib.import_module('dataset_mnist')
    dh = dataset.DataHandle()
    sample_input, sample_target = next_batch(dh)
    imsave(sample_input, sample_target, 0, train_dir)

    print("begin train...")
    for i in range(max_iter):
        batch = i + 1
        batch_input_track, batch_target_track = next_batch(dh)
        Loss, _ = sess.run([loss, opt], feed_dict={X: batch_input_track, Y: batch_target_track})
        str = 'batch[%d] loss[%3.3f]' % (batch, Loss)
        print str
        out.write(str + '\n')

        if batch % 10 == 1:
            sample_gen_img = sess.run(gen_img, {X: sample_input})
            imsave(sample_input, sample_gen_img, batch, train_dir)

        if i % 50 == 1:
            save(checkpoint_dir, saver, sess, batch)

def main(argv=None):
    _train()

if __name__ == '__main__':
    tf.app.run()





