import base64
import json
import numpy as np
import os
import tensorflow as tf

from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from io import BytesIO
from PIL import Image

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  def reframe_box_masks_to_image_masks_default():
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
      [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
      image=box_masks_expanded,
      boxes=reverse_boxes,
      box_ind=tf.range(num_boxes),
      crop_size=[image_height, image_width],
      extrapolation_value=0.0)
  image_masks = tf.cond(
    tf.shape(box_masks)[0] > 0,
    reframe_box_masks_to_image_masks_default,
    lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = reframe_box_masks_to_image_masks(
          detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
          tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
          detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

# Load the model
def init():
  global detection_graph

  # Retrieve the path to the model file using the model name
  model_path = Model.get_model_path('frozen_inference_graph.pb')
  
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # Setup data collectors
  global inputs_dc, prediction_dc
  inputs_dc = ModelDataCollector('best_model', designation='inputs', feature_names=['image'])
  prediction_dc = ModelDataCollector('best_model', designation='predictions', feature_names=['num_detections', 'detection_classes', 'detection_boxes', 'detection_scores'])

# Passes data to the model and returns the prediction
@rawhttp
def run(request):
  body = request.get_data(False)
  image = Image.open(BytesIO(body))
  image_np = load_image_into_numpy_array(image)

  output_dict = run_inference_for_single_image(image_np, detection_graph)
  num_detections = output_dict['num_detections']
  detection_classes = output_dict['detection_classes'].tolist()
  detection_boxes = output_dict['detection_boxes'].tolist()
  detection_scores = output_dict['detection_scores'].tolist()

  output = {}
  output['num_detections'] = num_detections
  output['detection_classes'] = detection_classes
  output['detection_boxes'] = detection_boxes
  output['detection_scores'] = detection_scores
  if 'detection_masks' in output_dict:
    output['detection_masks'] = output_dict['detection_masks'].tolist()

  inputs_dc.collect([base64.b64encode(body)])
  prediction_dc.collect([num_detections, json.dumps(detection_classes), json.dumps(detection_boxes), json.dumps(detection_scores)])
  
  return output
