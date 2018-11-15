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
import sys
import os
import numpy as np
import PIL
from PIL import Image, ImageDraw
import time

import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores
from src.region_to_bbox import region_to_bbox
from src.siam_mcf.siam_mcf_tracker import SiamMcfTracker
from src import vot


def track_one_sequence(hp,
                       design,
                       frame_name_list,
                       pos_x,
                       pos_y,
                       target_w,
                       target_h,
                       final_score_sz,
                       filename,
                       image,
                       templates_x,
                       templates_z,
                       scores_list,
                       vid_name,
                       dataset_type,
                       sess,
                       visualize_results,
                       save_images,
                       save_bboxes,
                       vot_handle,
                       gt=None):
    """ Handles tracking for one whole sequence. Inputs are fed to the network
    and the results are collected and can be shown on the screen and saved to
    the disk.

    Args:
      hp: namespace: hyperparameters.
      design: namespace: design parameters.
      frame_name_list: string list: list of sorted image paths to be read.
      pos_x: int: horizontal center of the target.
      pos_y: int: vertical center of the target.
      target_w: int: target width.
      target_h: int: target height.
      final_score_sz: int: size of the score map after upsampling.
      filename: string tensor: placeholder for the image path to be read.
      image: 3D tensor: the image read from the path.
      templates_x: 4D tensor: instance features from one or more layers
        concatenated by channels. See siam_mcf_net.inference comments for more
        details.
      templates_z: 4D tensor: exemplar features from one or more layers
        concatenated by channels. See siam_mcf_net.inference comments for more
        details.
      scores_list: 5D tensor: batch of score heatmaps for each of the selected
        layers.
      vid_name: string: name of this sequence (only for saving purposes).
      dataset_type: string: name of this dataset (only for saving purposes).
      sess: an open tf.Session to execute the graph.
      visualize_results: boolean: whether to show the results on the screen.
      save_images: boolean: whether to save image results to the disk.
      save_bboxes: boolean: whether to save bounding boxes to the disk.
      vot_handle: vot handle for running the VOT toolkit.
      gt: Nx4 array: optional ground truth bounding boxes (only for
        visualization purposes).

    Returns:
      Nx4 array: the resulting bounding boxes from the tracking.
      float: the tracking speed in frames per second.
    """
    num_frames = np.size(frame_name_list)
    # stores tracker's output for evaluation
    bboxes = np.zeros((num_frames, 4))

    if save_images:
        res_dir = 'results/%s/frames/%s' % (
            dataset_type, vid_name)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

    if save_bboxes:
        bb_res_dir = 'results/%s/bboxes' % (dataset_type)
        if not os.path.exists(bb_res_dir):
            os.makedirs(bb_res_dir)

    # save first frame position (from ground-truth)
    bboxes[0, :] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h

    if vot_handle is not None:
        frame_path = vot_handle.frame()
    else:
        frame_path = frame_name_list[0]

    tracker = SiamMcfTracker(
        design.context, design.exemplar_sz, design.search_sz, hp.scale_step,
        hp.scale_num, hp.scale_penalty, hp.scale_lr, hp.window_influence,
        design.tot_stride, hp.response_up, final_score_sz, pos_x, pos_y,
        target_w, target_h, frame_path, sess, templates_z, filename)

    t_start = time.time()

    # Get an image from the queue
    for i in range(1, num_frames):
        if vot_handle is not None:
            frame_path = vot_handle.frame()
        else:
            frame_path = frame_name_list[i]

        if save_images or visualize_results:
            image_ = sess.run(image, feed_dict={filename: frame_path})

        bbox = tracker.track(
            frame_path, sess, templates_z, templates_x, scores_list, filename)

        # convert <cx,cy,w,h> to <x,y,w,h> and save output
        bboxes[i, :] = bbox

        if vot_handle is not None:
            vot_rect = vot.Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
            vot_handle.report(vot_rect)

        if visualize_results:
            show_frame(image_, bboxes[i, :], 1)

        if save_images:
            out_img = Image.fromarray(image_.copy().astype(np.uint8))
            out_draw = ImageDraw.Draw(out_img)

            if gt is not None:
                gt_rect = np.array(region_to_bbox(gt[i, :], False)).astype(
                    np.int32)
                gt_rect[2:] = gt_rect[:2] + gt_rect[2:]

            rect = bboxes[i].copy()
            rect[2:] = rect[:2] + rect[2:]
            rect = rect.astype(np.int32)

            pillow_version = [int(x) for x in PIL.__version__.split('.')]
            if (pillow_version[0] > 5 or
                    (pillow_version[0] == 5 and pillow_version[1] >= 3)):
                if gt is not None:
                    out_draw.rectangle(
                        [tuple(gt_rect[:2]), tuple(gt_rect[2:])],
                        outline=(0, 0, 255),
                        width=2)
                out_draw.rectangle(
                    [tuple(rect[:2]), tuple(rect[2:])],
                    outline=(255, 0, 0),
                    width=3)
            else:
                if gt is not None:
                    out_draw.rectangle(
                        [tuple(gt_rect[:2]), tuple(gt_rect[2:])],
                        outline=(0, 0, 255))
                out_draw.rectangle(
                    [tuple(rect[:2]), tuple(rect[2:])],
                    outline=(255, 0, 0))

            out_img.save(os.path.join(res_dir, '%05d.jpg' % (i + 1)))

    t_elapsed = time.time() - t_start
    speed = num_frames/t_elapsed

    if save_bboxes:
        with open(os.path.join(bb_res_dir, vid_name+'.txt'), 'w') as f:
            for bb in bboxes:
                f.write('%.02f,%.02f,%.02f,%.02f\n' % tuple(bb))

    return bboxes, speed
