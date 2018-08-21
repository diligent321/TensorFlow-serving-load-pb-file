from __future__ import print_function

import os
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf
from tensorflow.python.platform import gfile

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('export_path_base', './tmp_scene_pb_SavedModels', 'Working directory.')
#tf.app.flags.DEFINE_string("pb_file", './trained_models/final_graph_4.13_add_food_2.5k.pb', 'Working directory.')
tf.app.flags.DEFINE_string("pb_file", './trained_models/scenedetect_425_v2.pb', 'Working directory.')

FLAGS = tf.app.flags.FLAGS

def main(_):
  print("------------------ EXPORT STARTING -----------------")

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    model_filename =FLAGS.pb_file
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
  export_path_base = FLAGS.export_path_base
  export_path = os.path.join(
     tf.compat.as_bytes(export_path_base),
     tf.compat.as_bytes(str(FLAGS.model_version)))
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  '''
  feature_configs = {
              'x': tf.FixedLenFeature(shape=[1,3,1080,1080], dtype=tf.float32),
              'y': tf.FixedLenFeature(shape=[1, 3, 1080, 1080], dtype=tf.float32)
  }
  serialized_example = tf.placeholder(tf.string, name="tf_example")
  tf_example = tf.parse_example(serialized_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')
  y = tf.identity(tf_example['y'], name='y')
 '''
  tensor_info_x = tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name('import/input:0'))
  tensor_info_y = tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name('import/final_result:0'))

  prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'images': tensor_info_x},
          outputs={'outputs': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)) 
  '''
  predict_input = x
  predict_output = y
  predict_signature_def_map = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={
                tf.saved_model.signature_constants.PREDICT_INPUTS: predict_input
            },
            outputs={
                tf.saved_model.signature_constants.PREDICT_OUTPUTS: predict_output
            }
  )
  '''

  legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
  builder.add_meta_graph_and_variables(
      sess=sess,
      tags=[tf.saved_model.tag_constants.SERVING],
      #signature_def_map={
      #    "predict_images": predict_signature_def_map
      #},
      signature_def_map={
          'predict_images': prediction_signature,
      },
      legacy_init_op=legacy_init_op
  )
  builder.save()

  print("------------------ DONE EXPORTING -----------------")

if __name__ == '__main__':
  tf.app.run()
