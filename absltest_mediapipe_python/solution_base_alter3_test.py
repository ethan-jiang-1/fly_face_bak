# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mediapipe.python.solution_base."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from mediapipe.framework import calculator_pb2
from mediapipe.framework.formats import detection_pb2
from mediapipe.python import solution_base

CO_TEST_GRAPH_CONFIG0 = """
      input_stream: 'image_in'
      output_stream: 'image_out'
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:image_in'
        output_stream: 'IMAGE:transformed_image_in'
      }
      node {
        calculator: 'ImageTransformationCalculator'
        input_stream: 'IMAGE:transformed_image_in'
        output_stream: 'IMAGE:image_out'
      }
      """

def load_config():
    #return CO_TEST_GRAPH_CONFIG0
    import os 
    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    pbtxt_filename = os.sep.join([dir_root, "ref_tflite_models", "selfie_segmentation_cpu.pbtxt"])
    if not os.path.isfile(pbtxt_filename):
        raise ValueError("")

    with open(pbtxt_filename, "rt") as fin:
        content = fin.read()
    return content

class SolutionBaseAltTest(parameterized.TestCase):

  @parameterized.named_parameters(('graph_without_side_packets', load_config(), None))
  def test_solution_reset(self, text_config, side_inputs):
    print()
    print('text_config:\t', text_config)
    print('side_inputs:\t', side_inputs)
    config_proto = text_format.Parse(text_config,
                                     calculator_pb2.CalculatorGraphConfig())

    #input_image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    input_image = np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3)
    print(input_image.shape)
    with solution_base.SolutionBase(
        graph_config=config_proto, side_inputs=side_inputs) as solution:
        print(solution)
        for _ in range(20):
            outputs = solution.process(input_image)
            #self.assertTrue(np.array_equal(input_image, outputs.image_out))
            solution.reset()
    print()

if __name__ == '__main__':
  absltest.main()
