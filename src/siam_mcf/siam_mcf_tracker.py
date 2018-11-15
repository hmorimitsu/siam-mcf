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

import src.siamese as siam


class SiamMcfTracker(object):
    """ SiamMCF tracker.

        Initialize it as:
            tracker = SiamMcfTracker(...)

        Then at each frame call:
            bbox = tracker.track(...)

        To obtain the results.
    """
    def __init__(self,
                 context_sz,
                 exemplar_sz,
                 search_sz,
                 scale_step,
                 scale_num,
                 scale_penalty,
                 scale_lr,
                 window_influence,
                 tot_stride,
                 response_up,
                 final_score_sz,
                 pos_x,
                 pos_y,
                 target_w,
                 target_h,
                 image_filename,
                 sess,
                 templates_z,
                 filename):
        """
        Args:
          context_sz: float: amount of context (background) to be added to the
            image.
          exemplar_sz: int: size of the exemplar image.
          search_sz: int: size of the instance image.
          scale_step: float: how much one change in scale changes the image
            size.
          scale_num: int: number of scales to be used for multi-scale search.
          scale_penalty: float: penalty amount to be applied for scale changes.
          scale_lr: float: how much the new scale contributes to the new scale.
          window_influence: float: how much the hann window affects the
            prediction.
          tot_stride: int: output stride compared to input size.
          response_up: int: upsample rate for the score heatmap.
          final_score_sz: int: size of the score map after upsampling.
          pos_x: int: horizontal center of the target.
          pos_y: int: vertical center of the target.
          target_w: int: target width.
          target_h: int: target height.
          image_filename: string: path to the image to be read.
          sess: an open tf.Session to execute the graph.
          templates_z: 4D tensor: exemplar features from one or more layers
            concatenated by channels. See siam_mcf_net.inference comments for
            more details.
          filename: string tensor: placeholder for the image path to be read.
        """
        self.context_sz = context_sz
        self.exemplar_sz = exemplar_sz
        self.search_sz = search_sz
        self.scale_step = scale_step
        self.scale_num = scale_num
        self.scale_penalty = scale_penalty
        self.scale_lr = scale_lr
        self.window_influence = window_influence
        self.tot_stride = tot_stride
        self.response_up = response_up
        self.final_score_sz = final_score_sz
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.target_w = target_w
        self.target_h = target_h
        self.scale_factors = (scale_step**np.linspace(
            -np.ceil(scale_num/2), np.ceil(scale_num/2), scale_num))
        # cosine window to penalize large displacements
        hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
        penalty = np.transpose(hann_1d) * hann_1d
        self.penalty = penalty / np.sum(penalty)

        context = context_sz*(target_w+target_h)
        self.z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
        self.x_sz = (float(search_sz) / exemplar_sz *
                     np.sqrt(np.prod((target_w+context)*(target_h+context))))

        self.templates_z_ = self._extract_exemplar_features(
            pos_x, pos_y, target_w, target_h, image_filename, sess,
            templates_z, filename)

    def track(self,
              image_filename,
              sess,
              templates_z,
              templates_x,
              scores_list,
              filename):
        """ Performs the tracking for one frame.

        Args:
          image_filename: string: path to the image to be read.
          sess: an open tf.Session to execute the graph.
          templates_z: 4D tensor: exemplar features from one or more layers
            concatenated by channels. See siam_mcf_net.inference comments for
            more details.
          templates_x: 4D tensor: instance features from one or more layers
            concatenated by channels. See siam_mcf_net.inference comments for
            more details.
          scores_list: 5D tensor: batch of score heatmaps for each of the
            selected layers.
          filename: string tensor: placeholder for the image path to be read.

        Returns:
          4D vector: the bounding box of the estimated target position.
        """
        templates_x_ = self._extract_instance_features(
            self.pos_x, self.pos_y, self.x_sz, image_filename, sess,
            templates_x, filename)

        scores_list_ = sess.run(
            scores_list,
            feed_dict={
                templates_z: self.templates_z_,
                templates_x: templates_x_
            })

        mean_scores_ = np.mean(scores_list_[:4], axis=0)
        scores_ = mean_scores_[:, :, :, 0]

        # penalize change of scale
        scores_[0, :, :] = self.scale_penalty*scores_[0, :, :]
        scores_[2, :, :] = self.scale_penalty*scores_[2, :, :]

        # find scale with highest peak (after penalty)
        new_scale_id = np.where(scores_ >= np.max(scores_))[0][0]
        # select response with new_scale_id
        score_ = scores_[new_scale_id]
        score_ = score_ - np.min(score_)
        score_ = score_ / np.sum(score_)
        # apply displacement penalty
        score_ = ((1 - self.window_influence) * score_ +
                  self.window_influence * self.penalty)

        scaled_search_area = self.x_sz * self.scale_factors
        scaled_target_w = self.target_w * self.scale_factors
        scaled_target_h = self.target_h * self.scale_factors

        # update scaled sizes
        x_sz = ((1-self.scale_lr) * self.x_sz +
                self.scale_lr*scaled_search_area[new_scale_id])
        target_w = ((1-self.scale_lr) * self.target_w +
                    self.scale_lr*scaled_target_w[new_scale_id])
        target_h = ((1-self.scale_lr) * self.target_h +
                    self.scale_lr*scaled_target_h[new_scale_id])
        self.target_w = max(target_w, 10)
        self.target_h = max(target_h, 10)
        self.x_sz = np.maximum(x_sz, 40)

        self.pos_x, self.pos_y = self._update_target_position(
            self.pos_x,
            self.pos_y,
            score_,
            self.final_score_sz,
            self.tot_stride,
            self.search_sz,
            self.response_up,
            self.x_sz)

        # convert <cx,cy,w,h> to <x,y,w,h> and save output
        bb = [self.pos_x-self.target_w/2, self.pos_y-self.target_h/2, 
              self.target_w, self.target_h]
        bb = np.array(bb)

        return bb

    def _extract_exemplar_features(self,
                                   pos_x,
                                   pos_y,
                                   target_w,
                                   target_h,
                                   image_filename,
                                   sess,
                                   templates_z,
                                   filename):
        """ Extracts features for the exemplar image.

        Args:
          pos_x: int: horizontal center of the target.
          pos_y: int: vertical center of the target.
          target_w: int: target width.
          target_h: int: target height.
          image_filename: string: path to the image to be read.
          sess: an open tf.Session to execute the graph.
          templates_z: 4D tensor: exemplar features from one or more layers
            concatenated by channels. See siam_mcf_net.inference comments for
            more details.
          filename: string tensor: placeholder for the image path to be read.

        Returns:
          4D array: exemplar features.
        """
        context = self.context_sz*(target_w+target_h)
        z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))

        templates_z_ = sess.run(templates_z, feed_dict={
            siam.pos_x_ph: pos_x,
            siam.pos_y_ph: pos_y,
            siam.z_sz_ph: z_sz,
            filename: image_filename})

        return templates_z_

    def _extract_instance_features(self,
                                   pos_x,
                                   pos_y,
                                   x_sz,
                                   image_filename,
                                   sess,
                                   templates_x,
                                   filename):
        """ Extracts features for the instance image.

        Args:
          pos_x: int: horizontal center of the target.
          pos_y: int: vertical center of the target.
          x_sz: int: size of the region that should be cropped from the
            instance image.
          image_filename: string: path to the image to be read.
          sess: an open tf.Session to execute the graph.
          templates_x: 4D tensor: instance features from one or more layers
            concatenated by channels. See siam_mcf_net.inference comments for
            more details.
          filename: string tensor: placeholder for the image path to be read.

        Returns:
          4D array: instance features.
        """
        scaled_search_area = x_sz * self.scale_factors
        templates_x_ = sess.run(
            templates_x,
            feed_dict={
                siam.pos_x_ph: pos_x,
                siam.pos_y_ph: pos_y,
                siam.x_sz0_ph: scaled_search_area[0],
                siam.x_sz1_ph: scaled_search_area[1],
                siam.x_sz2_ph: scaled_search_area[2],
                filename: image_filename})

        return templates_x_

    def _update_target_position(self,
                                pos_x,
                                pos_y,
                                score,
                                final_score_sz,
                                tot_stride,
                                search_sz,
                                response_up,
                                x_sz):
        """ Computes the new target position in frame coordinates.

        Args:
          pos_x: int: horizontal center of the target.
          pos_y: int: vertical center of the target.
          score: 2D array: the score heatmap.
          final_score_sz: int: size of the score map after upsampling.
          tot_stride: int: output stride compared to input size.
          search_sz: int: size of the instance image.
          response_up: int: upsample rate for the score heatmap.
          x_sz: int: size of the region that should be cropped from the
            instance image.

        Returns:
          float: the new horizontal center of the target.
          float: the new vertical center of the target.
        """
        # find location of score maximizer
        p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
        # displacement from the center in search area final representation ...
        center = float(final_score_sz - 1) / 2
        disp_in_area = p - center
        # displacement from the center in instance crop
        disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
        # displacement from the center in instance crop (in frame coordinates)
        disp_in_frame = disp_in_xcrop * x_sz / search_sz
        # *position* within frame in frame coordinates
        pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
        return pos_x, pos_y
