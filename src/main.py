#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

from model import *

params = monodepth_parameters(
    encoder="resnet50",
    height=256,
    width=512,
    batch_size=8,
    num_threads=8,
    num_epochs=50,
    do_stereo=False,
    wrap_mode="border",
    use_deconv=True,
    alpha_image_loss=0.85,
    disp_gradient_loss_weight=0.1,
    lr_loss_weight=1,
    full_summary=False)

total_epoch = 50
left_image_dir = r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2'
right_image_dir = r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_3'

def get_pic_name():
    left_img_name = os.listdir(left_image_dir)
    right_img_name = os.listdir(right_image_dir)

    return left_img_name, right_img_name

path = r'D:\stereo\stereo\KITTI_data\2011_09_26'
all_dir = os.listdir(path)
left_file_dir = []
right_file_dir = []
for dir in all_dir:
    left_dir = os.path.join(path, dir, 'image_02', 'data')
    right_dir = os.path.join(path, dir, 'image_03', 'data')
    left_file_dir.extend([os.path.join(left_dir, left) for left in os.listdir(left_dir)])
    right_file_dir.extend([os.path.join(right_dir, right) for right in os.listdir(right_dir)])

left_dir = r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2'
right_dir = r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_3'
left_file_dir.extend([os.path.join(left_dir, left) for left in os.listdir(left_dir)])
right_file_dir.extend([os.path.join(right_dir, right) for right in os.listdir(right_dir)])

test_left_file_dir = [r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_00.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_01.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_02.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_03.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_04.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_05.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_06.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_07.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_08.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_09.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_10.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_11.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_12.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_13.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_14.png',
                      r'D:\stereo\stereo\KITTI_data\data_scene_flow_multiview\training\image_2\000000_15.png']

def post_process_disparity(disp):
    h, w, _ = disp.shape
    l_disp = disp[:,:, 0]
    r_disp = np.fliplr(disp[:,:, 1])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

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

            '''
            left_batch, right_batch = get_batch_pic(0, batch_size, left_file_dir, right_file_dir)
            disp = post_process_disparity(((model.disp_left_est[0]).eval(feed_dict={model.left_pic: left_batch, model.right_pic: right_batch})).squeeze())
            min_disp = np.min(disp)
            max_disp = np.max(disp)
            new_disp = ((disp - min_disp) / (max_disp - min_disp) * 255).astype(np.uint8)
            cv.imshow("new_disp", cv.resize(new_disp, (1242, 375), interpolation=cv.INTER_CUBIC))
            cv.imshow("left_batch", cv.resize(left_batch[1], (1242, 375), interpolation=cv.INTER_CUBIC))
            cv.imshow("right_batch", cv.resize(right_batch[1], (1242, 375), interpolation=cv.INTER_CUBIC))
            cv.waitKey(0)
            '''
            save_path = saver.save(sess, './tmp/model_ver0.1')

def test(params, batch_size, height, width):
    left_data = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))
    right_data = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))

    model = Monocular_model(params, 'test', left_data, right_data)

    with tf.Session() as sess:

        steps = len(test_left_file_dir) // batch_size

        saver = tf.train.Saver()
        saver.restore(sess, '../tmp/model_ver0.1')


        for iteration in range(steps):
            start1 = time.clock()
            left_batch = get_batch_pic(iteration, batch_size, test_left_file_dir, None, "test")

            start = time.clock()
            outputs = np.array(((model.disp_est[0]).eval(feed_dict={model.left_pic: left_batch})).squeeze())
            disp_pp = post_process_disparity(outputs)


            end = time.clock()
            print("time:  ", end - start)
            min_disp = np.min(disp_pp)
            max_disp = np.max(disp_pp)
            new_disp = ((disp_pp - min_disp) / (max_disp - min_disp) * 255).astype(np.uint8)

            new_disp = cv.resize(new_disp, (1242, 375), interpolation=cv.INTER_CUBIC)
            src_RGB = cv.applyColorMap(new_disp, cv.COLORMAP_JET)
            cv.imshow("new_disp", src_RGB)

            min_disp = np.min(outputs[:, :, 0])
            max_disp = np.max(outputs[:, :, 0])
            new_disp = ((outputs[:, :, 0] - min_disp) / (max_disp - min_disp) * 255).astype(np.uint8)

            new_disp = cv.resize(new_disp, (1242, 375), interpolation=cv.INTER_CUBIC)
            src_RGB = cv.applyColorMap(new_disp, cv.COLORMAP_JET)
            cv.imshow("new_disp2", src_RGB)
            cv.waitKey(0)


if __name__ == "__main__":
    test(params, 1, 256, 512)

    train(params, 4, 256, 512, 0.0001)
