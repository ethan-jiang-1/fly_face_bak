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

'''
node {
  calculator: "InferenceCalculator"
  input_stream: "TENSORS:input_tensors"
  output_stream: "TENSORS:detection_tensors"
  node_options: {
    [mediapipe.InferenceCalculatorOptions.ext] {
      model_path: "mediapipe/modules/face_detection/face_detection_full_range_sparse.tflite"
      delegate {
        xnnpack {}
      }
    }
  }
}
'''

CO_TEST_GRAPH_CONFIG0 = """
# MediaPipe graph to detect faces. (CPU input, and inference is executed on
# CPU.)
#
# It is required that "face_detection_full_range_sparse.tflite" is available at
# "mediapipe/modules/face_detection/face_detection_full_range_sparse.tflite"
# path during execution.
#
# EXAMPLE:
#   node {
#     calculator: "FaceDetectionFullRangeCpu"
#     input_stream: "IMAGE:image"
#     output_stream: "DETECTIONS:face_detections"
#   }

type: "FaceDetectionFullRangeCpu"

# CPU image. (ImageFrame)
input_stream: "IMAGE:image"

# Detected faces. (std::vector<Detection>)
# NOTE: there will not be an output packet in the DETECTIONS stream for this
# particular timestamp if none of faces detected. However, the MediaPipe
# framework will internally inform the downstream calculators of the absence of
# this packet so that they don't wait for it unnecessarily.
output_stream: "DETECTIONS:detections"

# Converts the input CPU image (ImageFrame) to the multi-backend image type
# (Image).
node: {
  calculator: "ToImageCalculator"
  input_stream: "IMAGE_CPU:image"
  output_stream: "IMAGE:multi_backend_image"
}

# Transforms the input image into a 192x192 tensor while keeping the aspect
# ratio (what is expected by the corresponding face detection model), resulting
# in potential letterboxing in the transformed image.
node: {
  calculator: "ImageToTensorCalculator"
  input_stream: "IMAGE:multi_backend_image"
  output_stream: "TENSORS:input_tensors"
  output_stream: "MATRIX:transform_matrix"
  node_options: {
    [mediapipe.ImageToTensorCalculatorOptions.ext] {
      output_tensor_width: 192
      output_tensor_height: 192
      keep_aspect_ratio: true
      output_tensor_float_range {
        min: -1.0
        max: 1.0
      }
      border_mode: BORDER_ZERO
    }
  }
}

# Runs a TensorFlow Lite model on CPU that takes an image tensor and outputs a
# vector of tensors representing, for instance, detection boxes/keypoints and
# scores.
node {
  calculator: "InferenceCalculator"
  input_stream: "TENSORS:input_tensors"
  output_stream: "TENSORS:detection_tensors"
  node_options: {
    [mediapipe.InferenceCalculatorOptions.ext] {
      model_path: "face_detection/face_detection_full_range_sparse.tflite"
    }
  }
}

# Performs tensor post processing to generate face detections.
node {
  calculator: "FaceDetectionFullRangeCommon"
  input_stream: "TENSORS:detection_tensors"
  input_stream: "MATRIX:transform_matrix"
  output_stream: "DETECTIONS:detections"
}
"""


class SolutionBaseAltTest(parameterized.TestCase):

  @parameterized.named_parameters(('graph_without_side_packets', CO_TEST_GRAPH_CONFIG0, None))
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
