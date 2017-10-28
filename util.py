import os
import cv2
import numpy as np

def imsave(input_track, target_track, idx, save_dir):
    [batch_size, seqlen, image_size, image_size, num_channel] = np.shape(input_track)
    for i in range(batch_size):
        img = np.zeros((image_size, image_size * (seqlen + 1), num_channel))
        for j in range(seqlen):
            img[:, image_size * j: image_size * (j + 1)] = input_track[i, j]
        img[:, image_size * seqlen: image_size * (seqlen + 1)] = target_track[i]
        img = np.array(img * 255).astype(np.uint8)

        filename = '%s_%02dst.jpg' % (idx, i)
        path = os.path.join(save_dir, filename)
        cv2.imwrite(path, img)

def save(checkpoint_dir, saver, sess, step):
    model_name = 'dfn.model'
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step = step)
