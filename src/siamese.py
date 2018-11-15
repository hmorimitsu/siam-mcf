###############################################################################
#
# Modified by Henrique Morimitsu from the code on
# https://github.com/torrvision/siamfc-tf
# written by Luca Bertinetto and Jack Valmadre.
#
# The modified code is licensed under the BSD-3-clause license described below.
# But it may also be subject to the original siamfc-tf license.
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

import tensorflow as tf
import numpy as np
import sys
import os.path
from src.crops import (extract_crops_z, extract_crops_x, pad_frame,
                       resize_images)
from src.siam_mcf import siam_mcf_net

pos_x_ph = tf.placeholder(tf.float64)
pos_y_ph = tf.placeholder(tf.float64)
z_sz_ph = tf.placeholder(tf.float64)
x_sz0_ph = tf.placeholder(tf.float64)
x_sz1_ph = tf.placeholder(tf.float64)
x_sz2_ph = tf.placeholder(tf.float64)


def build_tracking_graph(root_dir,
                         final_score_sz,
                         design,
                         env,
                         hp):
    """ Defines and builds the tracking graph.

    Args:
      root_dir: string: path to the root directory of this project.
      final_score_sz: int: size of the score map after upsampling.
      design: namespace: design parameters.
      env: namespace: environment parameters.
      hp: namespace: hyperparameters.

    Returns:
      string tensor: placeholder for the image path to be read.
      3D tensor: the image read from the path.
      4D tensor: instance features from one or more layers concatenated
        by channels. See siam_mcf_net.inference comments for more details.
      4D tensor: exemplar features from one or more layers concatenated
        by channels. See siam_mcf_net.inference comments for more details.
      5D tensor: batch of score heatmaps for each of the selected layers.
    """
    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    # Decode the image as a JPEG file, this will turn it into a Tensor
    image = tf.image.decode_jpeg(image_file, channels=3)
    image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)
    # image = image[:, :, ::-1]
    frame_sz = tf.shape(image)
    # used to pad the crops
    if design.pad_with_image_mean:
        avg_chan = tf.reduce_mean(image, axis=(0, 1), name='avg_chan')
    else:
        avg_chan = None
    # pad with if necessary
    frame_padded_z, npad_z = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph,
                                       z_sz_ph, avg_chan)
    frame_padded_z = tf.cast(frame_padded_z, tf.float32)
    # extract tensor of z_crops
    z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x_ph, pos_y_ph,
                              z_sz_ph, design.exemplar_sz)
    z_crops = tf.concat([z_crops for _ in range(hp.scale_num)], axis=0)

    frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph,
                                       x_sz2_ph, avg_chan)
    frame_padded_x = tf.cast(frame_padded_x, tf.float32)
    # extract tensor of x_crops (3 scales)
    x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph,
                              x_sz0_ph, x_sz1_ph, x_sz2_ph, design.search_sz)

    templates_z, templates_x, scores_list = _create_siamese(
        design, x_crops, z_crops, use_res_reduce=True)

    # upsample the score maps
    scores_up_list = [tf.image.resize_images(
        s, [final_score_sz, final_score_sz],
        method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
        for s in scores_list]
    scores_up_list = tf.stack(scores_up_list)
    scores_list = tf.stack(scores_list)
    scores_up_list = tf.reshape(scores_up_list, [
        scores_up_list.get_shape()[0], hp.scale_num,
        scores_up_list.get_shape()[2], scores_up_list.get_shape()[3],
        scores_up_list.get_shape()[4]])
    scores_list = tf.reshape(scores_list, [
        scores_list.get_shape()[0], hp.scale_num, scores_list.get_shape()[2],
        scores_list.get_shape()[3], scores_list.get_shape()[4]])

    return filename, image, templates_x, templates_z, scores_up_list


def _create_siamese(design, instances, exemplars, use_res_reduce):
    """ Creates the siamese tracking network.

    Args:
      design: namespace: design parameters.
      instances: 4D tensor: batch of instance images.
      exemplars: 4D tensor: batch of exemplar images.
      use_res_reduce: boolean: whether to use residual adaptation modules in
        the output of each selected layer.

    Returns:
      4D tensor: instance features from one or more layers concatenated
        by channels. See siam_mcf_net.inference comments for more details.
      4D tensor: exemplar features from one or more layers concatenated
        by channels. See siam_mcf_net.inference comments for more details.
      5D tensor: batch of score heatmaps for each of the selected layers.
    """
    instances.set_shape([
        instances.get_shape()[0].value,
        instances.get_shape()[1].value,
        instances.get_shape()[2].value, 3])
    exemplars.set_shape([
        exemplars.get_shape()[0].value,
        exemplars.get_shape()[1].value,
        exemplars.get_shape()[2].value, 3])

    exe_features, inst_features = siam_mcf_net.inference(
        exemplars, instances, design.exemplar_output_stride,
        design.instance_output_stride, design.xcorr_layers,
        design.xcorr_channels, design.xcorr_exe_crop_sizes, False, 
        use_res_reduce)

    xcorr_list = siam_mcf_net.match_templates(
        exe_features, inst_features, design.xcorr_layers,
        design.xcorr_channels, design.exemplar_output_stride,
        design.instance_output_stride, False)

    return exe_features, inst_features, xcorr_list
