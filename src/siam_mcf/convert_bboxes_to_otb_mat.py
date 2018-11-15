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

import os
import numpy as np
import scipy.io as sio
import sys


def main(argv):
    """ Utility code to convert the text bounding box results from SiamMCF
    tracker into OTB MAT format.
    """
    if len(argv) < 4:
        print('Usage:')
        print('    python convert_bboxes_to_otb_mat.py [tracker_name] ' +
              '[results_dir_path] [otb_dataset_path] [output_dir_path]')
        sys.exit()
    tracker_name = argv[0]
    res_dir_path = argv[1]
    otb_dir_path = argv[2]
    out_dir_path = argv[3]

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    sequence_names = [f[:-4] for f in os.listdir(res_dir_path)
                      if f.endswith('.txt')]
    for seq_name in sequence_names:
        res = np.genfromtxt(os.path.join(res_dir_path, seq_name+'.txt'),
                            delimiter=',')
        gts = np.genfromtxt(os.path.join(
            otb_dir_path, seq_name, 'groundtruth_rect.txt'), delimiter=' ')
        assert len(res) == len(gts), (
            '%s results and GTs have different lengths (%d vs %d)' %
            (seq_name, len(res), len(gts)))
        out_mat_cell = generate_otb_mat_cell(res, gts)
        out_file_name = '%s_%s.mat' % (seq_name.lower(), tracker_name.lower())
        sio.savemat(os.path.join(out_dir_path, out_file_name),
                    mdict={'results': out_mat_cell})
        print('Saved: ' + out_file_name)


def generate_otb_mat_cell(res, gts):
    """ Creates the MAT content.

    Args:
      res: Nx4 array: predicted bounding boxes.
      gts: Nx4 array: ground truth bounding boxes.

    Returns:
      A MAT cell to be saved.
    """
    attr_dict = {}
    attr_dict['res'] = res
    attr_dict['type'] = 'rect'
    attr_dict['fps'] = 1.0
    attr_dict['len'] = len(res)
    attr_dict['annoBegin'] = 1
    attr_dict['startFrame'] = 1
    attr_dict['anno'] = gts
    res_cell = np.zeros([1], np.object)
    res_cell[0] = attr_dict
    return res_cell


if __name__ == '__main__':
    main(sys.argv[1:])
