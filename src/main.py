#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
from model import *


params = {}
params['alpha_image_loss'] = 0.85
params['do_stereo'] = True
params['disp_gradient_loss_weight'] = 0.1
params['lr_loss_weight'] = 1.0
params['pic_height'] = 375
params['pic_width'] = 1242

total_epoch = 10

def get_file(left_dir, right_dir, test_dir):
    left_file_dir = [os.path.join(left_dir, left) for left in os.listdir(left_dir)]
    right_file_dir= [os.path.join(right_dir, right) for right in os.listdir(right_dir)]
    test_left_file_dir = [os.path.join(test_dir, test) for test in os.listdir(test_dir)]

    return left_file_dir, right_file_dir, test_left_file_dir


def get_batch_pic(index, batch_size, left_img_name, right_img_name, mode):
    left_pic = []
    right_pic = []
    if mode == "train":
        for i in range(batch_size):
            left_pic.append(cv.resize(cv.imread(left_img_name[index * batch_size + i]),
                                      (512, 256), interpolation=cv.INTER_CUBIC))
            right_pic.append(cv.resize(cv.imread(right_img_name[index * batch_size + i]),
                                       (512, 256), interpolation=cv.INTER_CUBIC))

        return np.array(left_pic), np.array(right_pic)

    else:
        for i in range(batch_size):
            data = cv.imread(left_img_name[index * batch_size + i])
            left_pic.append(cv.resize(data,
                                      (512, 256), interpolation=cv.INTER_CUBIC))

        return np.array(left_pic)


def train(params, batch_size, height, width, learning_rate):
    left_data = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))
    right_data = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))

    model = Monocular_model(params, 'train', left_data, right_data)

    with tf.Session() as sess:
        loss = model.total_loss
        opt_step = tf.train.AdamOptimizer(learning_rate)
        grads = opt_step.compute_gradients(loss)

        global_step = tf.Variable(0, trainable=False)
        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        init.run()

        steps = len(left_file_dir) // batch_size

        for epoch in range(total_epoch):
            for i in range(steps):
                left_batch, right_batch = get_batch_pic(i, batch_size, left_file_dir, right_file_dir, 'train')
                _, loss_value = sess.run([apply_gradient_op, loss], feed_dict={model.left_pic: left_batch, model.right_pic: right_batch})
                print("epoch:", epoch, "loss_value:", loss_value)

            save_path = saver.save(sess, '../tmp/model_ver0.3')

def test(params, batch_size, height, width):
    left_data = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))
    right_data = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))

    model = Monocular_model(params, 'test', left_data, right_data)

    with tf.Session() as sess:

        steps = len(test_left_file_dir) // batch_size

        saver = tf.train.Saver()
        saver.restore(sess, '../tmp/model_ver0.3')

        for iteration in range(steps):
            left_batch = get_batch_pic(iteration, batch_size, test_left_file_dir, None, "test")
            outputs = np.array(((model.disp_est[0]).eval(feed_dict={model.left_pic: left_batch})).squeeze())

            min_disp = np.min(outputs[:, :, 0])
            max_disp = np.max(outputs[:, :, 0])
            new_disp = ((outputs[:, :, 0] - min_disp) / (max_disp - min_disp) * 255).astype(np.uint8)

            new_disp = cv.resize(new_disp, (params['pic_width'], params['pic_height']), interpolation=cv.INTER_CUBIC)
            src_RGB = cv.applyColorMap(new_disp, cv.COLORMAP_JET)
            cv.imshow("new_disp2", src_RGB)
            cv.imshow("new_disp3", new_disp)
            cv.waitKey(0)

if __name__ == "__main__":

    left_file_dir, right_file_dir, test_left_file_dir = get_file(r'../pic/left', r'../pic/right', r'../pic/test')
    train(params, 4, 256, 512, 0.0001)
    test(params, 1, 256, 512)

