# TensorFlow-serving-load-pb-file
In deploying the model serving, several model services can be provided on a single machine(usually just a cpu server), such as photo classification, scene detection and so on. In order to easily switch between several pretrained model, the frozen pb file(with weights parameter) can be generated on a gpu server, then send all pb files to the cpu server. Finally, the cpu server can be used to serve the http requests from mobile devices, the details are as follows.

Step 1: Installing ModelServer following official tutorial, https://www.tensorflow.org/serving/setup ;

Step 2:convert the checkpoints file to pb file， which comprises the model graph and weight parameters;

Step 3: Run the file export.py, convert the above pb file to SavedModel which is supported by tensorflow serving;

Step 4: Run the following command, and start the model server.(here, tmp_scene_pb_SavedModels is supposed to be the directory with SavedModel included.)
        
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mobilenet_v1 --model_base_path=/home/lfyu/official-tf-serving/tmp_scene_pb_SavedModels
        
Step 5: Run the following command, and start the http serving, which can parse the received bytes and return the class which is predicted by the pb model;

python do_inference-serving.py
