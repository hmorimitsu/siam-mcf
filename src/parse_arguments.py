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

import argparse
import json
import os
import sys
from collections import namedtuple


def parse_arguments(root_dir):
    """ Parse arguments from JSON parameter files.

    Args:
      root_dir: string: path to the directory containing the JSON files.

    Returns:
      namedtuple: hyperparameters.
      namedtuple: evaluation parameters.
      namedtuple: environment parameters.
      namedtuple: design parameters.
    """
    with open(os.path.join(
            root_dir, 'parameters/hyperparams.json')) as json_file:
        hp = json.load(json_file)
    with open(os.path.join(
            root_dir, 'parameters/evaluation.json')) as json_file:
        evaluation = json.load(json_file)
    with open(os.path.join(
            root_dir, 'parameters/environment.json')) as json_file:
        env = json.load(json_file)
    with open(os.path.join(
            root_dir, 'parameters/design_mcf.json')) as json_file:
        design = json.load(json_file)

    hp = namedtuple('hp', hp.keys())(**hp)
    evaluation = namedtuple('evaluation', evaluation.keys())(**evaluation)
    env = namedtuple('env', env.keys())(**env)
    design = namedtuple('design', design.keys())(**design)

    return hp, evaluation, env, design


def parse_command_line_arguments():
    """ Parse arguments from the command line.

    Returns:
      argparse.Namespace: arguments read from the command line.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name',
        help=('Name of the dataset directory to read the sequences from. ' +
              'It must be the same name as the directory name in data/'),
        default='vot16'
    )
    parser.add_argument(
        '--save_bboxes',
        help='If true, saves bounding box results to disk.',
        action='store_true'
    )
    parser.add_argument(
        '--save_images',
        help='If true, saves image results to disk.',
        action='store_true'
    )
    parser.add_argument(
        '--visualize',
        help='If true, displays results on the screen.',
        action='store_true'
    )
    args = parser.parse_args()

    return args
