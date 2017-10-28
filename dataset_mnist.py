import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

class DataHandle(object):
    def __init__(self, mnistDataset = 'datasets/mnist.h5', mode = 'standard', background='zeros', num_frames = 3, batch_size = 16, image_size = 64, num_diagit=2, step_length = 0.1):
        self.mode_ = mode
        self.background_ = background
        self.seq_length_ = num_frames
        self.image_size_ = image_size
        self.num_digits_ = num_diagit
        self.step_length_ = step_length
        self.batch_size_ = batch_size
        self.dataset_size_ = 10000
        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2
        self.num_channels_ = 1

        f = h5py.File(mnistDataset)
        self.data_ = f['train'].value.reshape(-1,28,28)
        f.close()

        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        np.random.shuffle(self.indices_)

    def GetBatchSize(self):
        return self.batch_size_

    def GetDims(self):
        return self.frame_size_

    def GetDatasetSize(self):
        return self.dataset_size_

    def GetSeqLength(self):
        return self.seq_length_

    def Reset(self):
        pass

    def GetRandomTrajectory(self, batch_size):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_

        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        theta = np.random.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        for i in xrange(length):

            y += v_y * self.step_length_
            x += v_x * self.step_length_

            for j in xrange(batch_size):
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1:
                    y[j] = 1
                    v_y[j] = -v_y[j]
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1:
                    x[j] = 1
                    v_x[j] = -v_x[j]
            start_x[i, :] = x
            start_y[i, :] = y

        start_x = (start_x * canvas_size).astype(np.int32)
        start_y = (start_y * canvas_size).astype(np.int32)
        return start_y, start_x

    def Overlap(self, a, b):
        return np.maximum(a, b)

    def GetBatch(self):
        start_y, start_x = self.GetRandomTrajectory(self.batch_size_ * self.num_digits_)

        if self.background_ == 'zeros':
            data = np.zeros((self.batch_size_, self.num_channels_, self.image_size_, self.image_size_, self.seq_length_), dtype = np.float32)
        elif self.background_ == 'rand':
            data = np.random.rand(self.batch_size_, self.num_channels_, self.image_size_, self.image_size_, self.seq_length_)

        for j in xrange(self.batch_size_):
            for n in xrange(self.num_digits_):
                ind = self.indices_[self.row_]
                self.row_ += 1
                if self.row_ == self.data_.shape[0]:
                    self.row_ = 0
                    np.random.shuffle(self.indices_)
                digit_image = self.data_[ind]
                digit_size = self.digit_size_

                if self.mode_ == 'squares':
                    digit_size = np.random.randint(5,20)
                    digit_image = np.ones((digit_size, digit_size), dtype=np.float32)

                for i in xrange(self.seq_length_):
                    top = start_y[i, j * self.num_digits_ + n]
                    left  = start_x[i, j * self.num_digits_ + n]
                    bottom = top + digit_size
                    right = left + digit_size
                    data[j,:,top: bottom, left: right, i] = self.Overlap(digit_image, data[j, :, top: bottom, left: right, i])

        return data
