# TensorFlow-serving-load-pb-file
In deploying the model serving, several model services can be provided on a single machine(usually just a cpu server), such as photo classification, scene detection and so on. In order to easily switch between several pretrained model, the frozen pb file(with weights parameter) can be generated on a gpu server, then send all pb files to the cpu server. Finally, the cpu server can be used to serve the http requests from mobile devices, the details are as follows.

Step 1: convert the checkpoints file to pb file， which comprises the model graph and weight parameters;
Step 2:
