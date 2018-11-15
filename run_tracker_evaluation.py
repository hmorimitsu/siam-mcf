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

from __future__ import division
import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import src.siamese as siam
from src.track_one_sequence import track_one_sequence
from src.parse_arguments import parse_arguments, parse_command_line_arguments
from src.region_to_bbox import region_to_bbox

root_dir = os.path.abspath(os.path.dirname(__file__))


def main(argv):
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    hp, evaluation, env, design = parse_arguments(root_dir)
    cmd_args = parse_command_line_arguments()

    if 'otb13' in cmd_args.dataset_name:
        dataset_type = 'otb13'
    elif 'otb15' in cmd_args.dataset_name:
        dataset_type = 'otb15'
    elif 'vot16' in cmd_args.dataset_name:
        dataset_type = 'vot16'
    elif 'vot17' in cmd_args.dataset_name:
        dataset_type = 'vot17'

    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    filename, image, templates_x, templates_z, scores_list =\
        siam.build_tracking_graph(
            root_dir, final_score_sz, design, env, hp)

    # iterate through all videos of dataset_name
    videos_folder = os.path.join(
        root_dir, env.root_dataset, cmd_args.dataset_name)
    videos_list = [v for v in os.listdir(videos_folder)
                   if os.path.isdir(os.path.join(videos_folder, v))]
    videos_list.sort()
    nv = np.size(videos_list)
    speed = np.zeros(nv * evaluation.n_subseq)
    precisions = np.zeros(nv * evaluation.n_subseq)
    precisions_auc = np.zeros(nv * evaluation.n_subseq)
    ious = np.zeros(nv * evaluation.n_subseq)
    lengths = np.zeros(nv * evaluation.n_subseq)
    successes = np.zeros(nv * evaluation.n_subseq)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        vars_to_load = []
        for v in tf.global_variables():
            if 'postnorm' not in v.name:
                vars_to_load.append(v)

        siam_ckpt_name = 'pretrained/siam_mcf.ckpt-50000'
        siam_saver = tf.train.Saver(vars_to_load)
        siam_saver.restore(sess, siam_ckpt_name)

        for i in range(nv):
            gt, frame_name_list, frame_sz, n_frames = _init_video(
                videos_list[i], videos_folder, dataset_type)
            starts = np.rint(np.linspace(
                0, n_frames - 1, evaluation.n_subseq + 1))
            starts = starts[0:evaluation.n_subseq]
            for j in range(evaluation.n_subseq):
                start_frame = int(starts[j])
                gt_ = gt[start_frame:, :]
                frame_name_list_ = frame_name_list[start_frame:]
                pos_x, pos_y, target_w, target_h = region_to_bbox(gt_[0])
                idx = i * evaluation.n_subseq + j
                bboxes, speed[idx] = track_one_sequence(
                    hp, design, frame_name_list_, pos_x, pos_y,
                    target_w, target_h, final_score_sz, filename,
                    image, templates_x, templates_z, scores_list,
                    videos_list[i], dataset_type, sess, cmd_args.visualize,
                    cmd_args.save_images, cmd_args.save_bboxes,
                    vot_handle=None, gt=gt_)
                (lengths[idx], precisions[idx], precisions_auc[idx], ious[idx],
                 successes[idx]) = _compile_results(
                     gt_, bboxes, evaluation.dist_threshold)
                print(str(i) + ' -- ' + videos_list[i] +
                      ' -- Precision: ' + "%.2f" % precisions[idx] +
                      ' -- Precisions AUC: ' + "%.2f" % precisions_auc[idx] +
                      ' -- IOU: ' + "%.2f" % ious[idx] +
                      ' -- Success@0.5: ' + "%.2f" % successes[idx] +
                      ' -- Speed: ' + "%.2f" % speed[idx] + ' --')

    tot_frames = np.sum(lengths)
    mean_precision = np.sum(precisions * lengths) / tot_frames
    mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
    mean_iou = np.sum(ious * lengths) / tot_frames
    mean_speed = np.sum(speed * lengths) / tot_frames
    mean_success = np.sum(successes * lengths) / tot_frames
    print('-- Overall stats (averaged per frame) on ' + str(nv) +
          ' videos (' + str(tot_frames) + ' frames) --')
    print(' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' +
          '%.2f' % mean_precision +
          ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc +
          ' -- IOU: ' + "%.2f" % mean_iou +
          ' -- Success@0.5: ' + "%.2f" % mean_success +
          ' -- Speed: ' + "%.2f" % mean_speed + ' --')


def _compile_results(gt, bboxes, dist_threshold):
    """ Computes the results for one sequence based on the tracking bounding
    boxes.

    Args:
      gt: Nx4 array: ground truth bounding boxes.
      bboxes: Nx4 array: predicted bounding boxes.
      dist_threshold: int: threshold in pixels to calculate the precision.

    Returns:
      int: number of boxes/frames in the sequence.
      float: precision of the results.
      float: precision AuC of the results.
      float: IoU of the results.
      float: success rate of the results.
    """
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior
    # to given threshold? (OTB metric)
    precision = (sum(new_distances < dist_threshold) /
                 np.size(new_distances) * 100)

    success = sum(new_ious > 0.5)/np.size(new_ious) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = (sum(new_distances < thresholds[i]) /
                             np.size(new_distances))

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou, success


def _init_video(video, videos_folder, dataset_type):
    """ Collects data for one sequence.

    Args:
      video: string: name of the sequence.
      videos_folder: string: path to the directory where the sequences are.
      dataset_type: string: supported types are 'otb' and 'vot'.

    Returns:
      Nx4 array: ground truth bounding boxes.
      string list: sorted list with the paths to the frames of the sequence.
      2d vector: size of the image.
      int: number of frames in the sequence.
    """
    vid_folder = os.path.join(videos_folder, video)
    if 'vot' in dataset_type:
        frame_name_list = [os.path.join(vid_folder, f)
                           for f in os.listdir(vid_folder)
                           if f.endswith(".jpg")]
    elif 'otb' in dataset_type:
        frame_name_list = [os.path.join(vid_folder, 'img', f)
                           for f in os.listdir(os.path.join(vid_folder, 'img'))
                           if f.endswith(".jpg")]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    if 'otb' in dataset_type:
        gt_file = os.path.join(vid_folder, 'groundtruth_rect.txt')
        gt = np.genfromtxt(gt_file, delimiter=' ')
        if len(gt.shape) != 2 or gt.shape[1] != 4:
            gt = np.genfromtxt(gt_file, delimiter=',')
            if len(gt.shape) != 2 or gt.shape[1] != 4:
                gt = np.genfromtxt(gt_file, delimiter='\t')
    elif 'vot' in dataset_type:
        gt_file = os.path.join(vid_folder, 'groundtruth.txt')
        gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), ('Number of frames and number of GT lines ' +
                                 'should be equal.')

    return gt, frame_name_list, frame_sz, n_frames


def _compute_distance(boxA, boxB):
    """ Computes the center distance between two bounding boxes.

    Args:
      boxA: 4D vector: first bounding box
      boxB: 4D vector: second bounding box

    Returns:
      float: the cener distance between both bounding boxes.
    """
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    """ Computes the IoU between two bounding boxes.

    Args:
      boxA: 4D vector: first bounding box
      boxB: 4D vector: second bounding box

    Returns:
      float: the IoU between both bounding boxes.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
