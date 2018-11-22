#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from collections import namedtuple
import cv2 as cv


class Monocular_model(object):
    def __init__(self, params, mode, left_pic, right_pic):
        self.params = params
        self.mode = mode
        self.left_pic = left_pic
        self.right_pic = right_pic

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.total_loss()

    def build_decoder(self):
        with tf.name_scope("decoder"):
            upconv_layer6 = self.upconv(self.conv_layer5, 512, 3, 2)
            concat_layer6 = tf.concat([upconv_layer6, self.skip5], 3)
            iconv_layer6 = self.conv(concat_layer6, 512, 3, 1)

            upconv_layer5 = self.upconv(iconv_layer6, 256, 3, 2)
            concat_layer5 = tf.concat([upconv_layer5, self.skip4], 3)
            iconv_layer5 = self.conv(concat_layer5, 256, 3, 1)

            upconv_layer4 = self.upconv(iconv_layer5, 128, 3, 2)
            concat_layer4 = tf.concat([upconv_layer4, self.skip3], 3)
            iconv_layer4 = self.conv(concat_layer4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv_layer4)
            udisp_layer4 = self.upsample_nn(self.disp4, 2)

            upconv_layer3 = self.upconv(iconv_layer4, 64, 3, 2)
            concat_layer3 = tf.concat([upconv_layer3, self.skip2, udisp_layer4], 3)
            iconv_layer3 = self.conv(concat_layer3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv_layer3)
            udisp_layer3 = self.upsample_nn(self.disp3, 2)

            upconv_layer2 = self.upconv(iconv_layer3, 32, 3, 2)
            concat_layer2 = tf.concat([upconv_layer2, self.skip1, udisp_layer3], 3)
            iconv_layer2 = self.conv(concat_layer2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv_layer2)
            udisp_layer2 = self.upsample_nn(self.disp2, 2)

            upconv_layer1 = self.upconv(iconv_layer2, 16, 3, 2)
            concat_layer1 = tf.concat([upconv_layer1, udisp_layer2], 3)
            iconv_layer1 = self.conv(concat_layer1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv_layer1)

    def build_encoder(self):
        with tf.name_scope('encoder'):
            self.conv_layer1 = self.conv(self.input, 64, 7, 2)
            self.pool_layer1 = self.maxpool(self.conv_layer1, 3)
            self.conv_layer2 = self.resnet_block(self.pool_layer1, 64, 3)
            self.conv_layer3 = self.resnet_block(self.conv_layer2, 128, 4)
            self.conv_layer4 = self.resnet_block(self.conv_layer3, 256, 6)
            self.conv_layer5 = self.resnet_block(self.conv_layer4, 512, 3)

        with tf.name_scope("skips"):
            self.skip1 = self.conv_layer1
            self.skip2 = self.pool_layer1
            self.skip3 = self.conv_layer2
            self.skip4 = self.conv_layer3
            self.skip5 = self.conv_layer4

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.name_scope('model'):

                if self.params['do_stereo']:
                    self.input = tf.concat([self.left_pic, self.right_pic], 3)
                else:
                    self.input = self.left_pic

                self.build_encoder()
                self.build_decoder()

                self.left_pyramid = self.pyramid(self.left_pic, 4)
                if self.mode == 'train':
                    self.right_pyramid = self.pyramid(self.right_pic, 4)

    def pyramid(self, org_img, scale_factor):
        scaled_imgs = [org_img]
        s = tf.shape(org_img)
        height, width = s[1], s[2]
        for i in range(scale_factor - 1):
            height //= 2
            width //= 2
            scaled_imgs.append(tf.image.resize_nearest_neighbor(org_img, [height, width]))
        return scaled_imgs

    def build_outputs(self):
        with tf.name_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]

        if self.mode == 'test':
            return

        with tf.name_scope("images"):
            self.left_est = [self.get_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]
            self.right_est = [self.get_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        with tf.name_scope("l-r"):
            self.right_to_left_disp = [self.get_image_left(self.disp_right_est[i], self.disp_left_est[i]) for i in
                                       range(4)]
            self.left_to_right_disp = [self.get_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in
                                       range(4)]

        with tf.name_scope("soomth"):
            self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

    def total_loss(self):
        with tf.name_scope('losses'):
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]


            self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]


            self.image_loss_right = [
                self.params['alpha_image_loss'] * self.ssim_loss_right[i] + (1 - self.params['alpha_image_loss']) *
                self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left = [
                self.params['alpha_image_loss'] * self.ssim_loss_left[i] + (1 - self.params['alpha_image_loss']) *
                self.l1_reconstruction_loss_left[i] for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)


            self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in
                                    range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)


            self.lr_left_loss = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in
                                 range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i
                                  in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)


            self.total_loss = self.image_loss + self.params['disp_gradient_loss_weight'] * self.disp_gradient_loss + self.params['lr_loss_weight'] * self.lr_loss

    def gradient_x(self, img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def gradient_y(self, img):
        return img[:, :-1, :, :] - img[:, 1:, :, :]

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]

        return smoothness_x + smoothness_y

    def get_image_left(self, img, disp):
        return bilinear_sampler(img, -disp)

    def get_image_right(self, img, disp):
        return bilinear_sampler(img, disp)

    def resnet_block(self, x, layers, blocks):
        for i in range(blocks - 1):
            for i in range(blocks - 1):
                out = self.resnet_conv(x, layers, 1)

        return self.resnet_conv(out, layers, 2)

    def resnet_conv(self, x, layers, stride):
        cut_flag = tf.shape(x)[3] != layers or stride == 2
        conv1 = self.conv(x, layers, 1, 1)
        conv2 = self.conv(conv1, layers, 3, stride)
        conv3 = self.conv(conv2, 4 * layers, 1, 1, None)

        if cut_flag:
            shortcut = self.conv(x, 4 * layers, 1, stride, None)
        else:
            shortcut = x

        return tf.nn.elu(conv3 + shortcut)

    def conv(self, x, output_chnnel, filter_size, stride, activation_fn=tf.nn.elu):
        return slim.conv2d(x, output_chnnel, filter_size, stride, 'SAME', activation_fn=activation_fn)

    def maxpool(self, x, filter_size):
        return slim.max_pool2d(x, filter_size, padding='SAME')

    def upconv(self, x, output_chnnel, filter_size, scale_factor):
        return slim.conv2d_transpose(x, output_chnnel, filter_size, scale_factor, 'SAME')

    def upconv_test(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def upsample_nn(self, x, ratio):
        return tf.image.resize_nearest_neighbor(x, [tf.shape(x)[1] * ratio, tf.shape(x)[2] * ratio])

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mean_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mean_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mean_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mean_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mean_x * mean_y

        SSIM = ((2 * mean_x * mean_y + C1) * (2 * sigma_xy + C2)) / \
               ((mean_x ** 2 + mean_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp


def bilinear_sampler(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.name_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.name_scope('_interpolate'):
            im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            x = x + 1
            y = y + 1

            x = tf.clip_by_value(x, 0.0, _width_f - 1 + 2 * 1)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f, _width_f - 1 + 2 * 1), tf.int32)

            dim2 = (_width + 2 * 1)
            dim1 = (_width + 2 * 1) * (_height + 2 * 1)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.name_scope('transform'):
            x_t, y_t = tf.meshgrid(tf.linspace(0.0, _width_f - 1.0, _width),
                                   tf.linspace(0.0, _height_f - 1.0, _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch = tf.shape(input_images)[0]
        _height = tf.shape(input_images)[1]
        _width = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f = tf.cast(_width, tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
        return output