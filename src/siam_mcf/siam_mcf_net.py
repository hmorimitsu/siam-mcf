###############################################################################
#
# Copyright (c) 2018, Henrique Morimitsu.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# #############################################################################

import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim

BN_CENTER = True
BN_SCALE = True
BN_DECAY = 0.997
BN_EPSILON = 1e-5


def inference(exemplars,
              instances,
              exemplar_output_stride,
              instance_output_stride,
              xcorr_layers,
              xcorr_channels,
              xcorr_exe_crop_sizes,
              is_training,
              use_res_reduce):
    """ Build the SiamMCF backbone network to extract exemplar and instance
    features.

    Args:
      exemplars: 4D tensor: batch of exemplar images.
      instances: 4D tensor: batch of instance images.
      exemplar_output_stride: int: output stride for the exemplar branch.
      instance_output_stride: int: output stride for the instance branch.
      xcorr_layers: string list: list of layer names to collect features from.
      xcorr_channels: int list: number of channels to be used in the output
        from each of the selected layers.
      xcorr_exe_crop_sizes: int list: size of the center square to be cropped
        from each of the extracted exemplar features.
      is_training: boolean: whether to run in training mode or not.
      use_res_reduce: boolean: whether to use residual adaptation modules in
        the output of each selected layer.

    Returns:
      4D tensor: the extracted exemplar features. They are concatenated by
        channel. Therefore, if we collected features from 4 different layers
        and each feature has 64 channels, this output will have 4*64=256
        channels.
      4D tensor: the extracted exemplar features. Uses the same concatenation
        rule as the exemplar.
    """
    MEAN_RGB = [123.68 / 255, 116.78 / 255, 103.94 / 255]
    exemplars = exemplars / 255.0
    exemplars = exemplars - tf.reshape(
        tf.constant(MEAN_RGB, tf.float32), [1, 1, 1, 3])
    instances = instances / 255.0
    instances = instances - tf.reshape(
        tf.constant(MEAN_RGB, tf.float32), [1, 1, 1, 3])

    with tf.variable_scope('siam') as sc:
        print('input sizes', exemplars.get_shape(), instances.get_shape())

        # Siamese ResNet50 network to extract features
        with slim.arg_scope(resnet_v2.resnet_arg_scope(
                batch_norm_decay=BN_DECAY,
                batch_norm_epsilon=BN_EPSILON,
                batch_norm_scale=BN_SCALE)):
            _, exe_end_points = resnet_v2.resnet_v2_50(
                exemplars,
                num_classes=None,
                is_training=is_training,
                global_pool=False,
                output_stride=exemplar_output_stride,
                reuse=False,
                scope=sc)
            _, inst_end_points = resnet_v2.resnet_v2_50(
                instances,
                num_classes=None,
                is_training=is_training,
                global_pool=False,
                output_stride=instance_output_stride,
                reuse=True,
                scope=sc)

    exe_outputs_list = []
    inst_outputs_list = []
    inter_padding = 'SAME'
    for ilayer in range(len(xcorr_layers)):
        exe_inter_net = exe_end_points[xcorr_layers[ilayer]]
        inst_inter_net = inst_end_points[xcorr_layers[ilayer]]

        if xcorr_channels[ilayer] > 0:
            # Residual adaptation modules
            if use_res_reduce:
                with tf.variable_scope(
                        'res_reduce_%d' % (ilayer + 1)) as sc_reduce:
                    with slim.arg_scope(resnet_v2.resnet_arg_scope(
                            batch_norm_decay=BN_DECAY,
                            batch_norm_epsilon=BN_EPSILON,
                            batch_norm_scale=BN_SCALE)):
                        rate = max(1, int(np.power(2.0, ilayer-1)))
                        exe_inter_net = resnet_v2.bottleneck(
                            exe_inter_net,
                            exe_inter_net.get_shape()[3].value,
                            exe_inter_net.get_shape()[3].value//4,
                            1,
                            rate=(rate, rate),
                            scope=sc_reduce)
                        sc_reduce.reuse_variables()
                        inst_inter_net = resnet_v2.bottleneck(
                            inst_inter_net,
                            inst_inter_net.get_shape()[3].value,
                            inst_inter_net.get_shape()[3].value//4,
                            1,
                            rate=(rate, rate),
                            scope=sc_reduce)

            # Additional batchnorm and convolution to reduce channels
            with tf.variable_scope('reduce_%d' % (ilayer + 1)) as sc_reduce:
                exe_inter_net = slim.batch_norm(
                    exe_inter_net,
                    decay=BN_DECAY,
                    epsilon=BN_EPSILON,
                    scale=BN_SCALE,
                    center=BN_CENTER,
                    activation_fn=tf.nn.relu,
                    is_training=is_training,
                    reuse=False,
                    scope=sc_reduce)
                exe_inter_net = slim.conv2d(
                    exe_inter_net,
                    xcorr_channels[ilayer],
                    [3, 3],
                    activation_fn=None,
                    padding=inter_padding,
                    reuse=False,
                    scope=sc_reduce)
                inst_inter_net = slim.batch_norm(
                    inst_inter_net,
                    decay=BN_DECAY,
                    epsilon=BN_EPSILON,
                    scale=BN_SCALE,
                    center=BN_CENTER,
                    activation_fn=tf.nn.relu,
                    is_training=is_training,
                    reuse=True,
                    scope=sc_reduce)
                inst_inter_net = slim.conv2d(
                    inst_inter_net,
                    xcorr_channels[ilayer],
                    [3, 3],
                    activation_fn=None,
                    padding=inter_padding,
                    reuse=True,
                    scope=sc_reduce)

        # Crop center area of exemplar features
        if xcorr_exe_crop_sizes[ilayer] > 0:
            top = ((exe_inter_net.get_shape()[1].value // 2) -
                   (xcorr_exe_crop_sizes[ilayer] // 2))
            left = ((exe_inter_net.get_shape()[2].value // 2) -
                    (xcorr_exe_crop_sizes[ilayer] // 2))
            exe_inter_net = exe_inter_net[
                :,
                top:top+xcorr_exe_crop_sizes[ilayer],
                left:left+xcorr_exe_crop_sizes[ilayer],
                :]

        exe_outputs_list.append(exe_inter_net)
        inst_outputs_list.append(inst_inter_net)

    # Concatenate by channel as opposed to creating a 5D tensor in order
    # to be able to concatenate features with different number of channels
    exe_concat_features = tf.concat(exe_outputs_list, axis=3)
    inst_concat_features = tf.concat(inst_outputs_list, axis=3)

    return exe_concat_features, inst_concat_features


def match_templates(exe_concat_features,
                    inst_concat_features,
                    xcorr_layers,
                    xcorr_channels,
                    exemplar_output_stride,
                    instance_output_stride,
                    is_training):
    """ Correlates the exemplar and instance features.

    Args:
      exe_concat_features: exemplar features from one or more layers
        concatenated by channels. See siam_mcf_net.inference comments for more
        details.
      exe_concat_features: instance features from one or more layers
        concatenated by channels. See siam_mcf_net.inference comments for more
        details.
      xcorr_layers: string list: list of layer names to collect features from.
      xcorr_channels: int list: number of channels to be used in the output
        from each of the selected layers.
      exemplar_output_stride: int: output stride for the exemplar branch.
      instance_output_stride: int: output stride for the instance branch.
      is_training: boolean: whether to run in training mode or not.

    Returns:
      4D tensor list: a list of the computed correlation heatmaps for each
        layer.
    """
    xcorr_list = []

    exe_concat_features = tf.split(exe_concat_features, xcorr_channels, axis=3)
    inst_concat_features = tf.split(inst_concat_features, xcorr_channels, axis=3)

    xcorr_padding = 'SAME'
    for ilayer in range(len(exe_concat_features)):
        print('xcorr %d from %s' % (
            ilayer + 1,
            xcorr_layers[ilayer]),
            exe_concat_features[ilayer].get_shape(),
            inst_concat_features[ilayer].get_shape())
        with tf.variable_scope('xcorr_%d' % (ilayer + 1)) as sc:
            single_exe = tf.split(
                exe_concat_features[ilayer],
                exe_concat_features[ilayer].get_shape()[0].value,
                axis=0)
            single_exe = [tf.expand_dims(x[0], axis=3) for x in single_exe]
            single_inst = tf.split(
                inst_concat_features[ilayer],
                inst_concat_features[ilayer].get_shape()[0].value,
                axis=0)

            rate = int(exemplar_output_stride / instance_output_stride)
            single_xcorr = [tf.nn.atrous_conv2d(i, e, rate, xcorr_padding)
                            for i, e in zip(single_inst, single_exe)]

            xcorr = tf.concat(single_xcorr, axis=0)

        with tf.variable_scope('adjust_%d' % (ilayer + 1)) as sc:
            xcorr = slim.batch_norm(
                xcorr,
                decay=BN_DECAY,
                epsilon=BN_EPSILON,
                center=BN_CENTER,
                scale=BN_SCALE,
                is_training=is_training,
                reuse=False,
                scope=sc)

        print('xcorr size', xcorr.get_shape())
        xcorr_list.append(xcorr)

    return xcorr_list
