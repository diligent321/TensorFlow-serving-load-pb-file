#-*- coding:utf-8 -*-

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import cv2
import time
from grpc._cython import cygrpc
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from scipy import misc

import skimage
from flask import Flask, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import skimage.io
import json

#config the sockets receive
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './tmp_upload'
#os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#app.config['SERVER_NAME'] = '0.0.0.0' + 'vision/recognize' + '80'

#ClassInd = ['food', 'text', 'scene', 'person', 'other', 'animal']
ClassInd = ['人像', '蓝天', '海滩', '冰雪', '美食', '花', '建筑', '狗', '猫','夜景', '日出日落', '街景', '文档', '舞台', '卖场', '烟花', '草地', '多肉', '盆栽', '商场', '绿植']
tf.app.flags.DEFINE_string('server', 'localhost:9000',
			   'PredictionService host:port')
tf.app.flags.DEFINE_string('server_out', '0.0.0.0:80',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('local_file_name', './samples/test_food.jpg',
		    """test image filename""")
tf.app.flags.DEFINE_integer('image_size', 160,
			   'resized image size of the model')
FLAGS = tf.app.flags.FLAGS

global_init_flag = 'True'               

def preprocess(f_handler, local_flag=0):
    def prewhiten(x):
        mean = 127.5
        std = 127.5
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y
    str_encode = f_handler.read()
    if local_flag==1:
        data = np.fromstring(str_encode, np.uint8)
        angle = 0
    else:
        angle = np.fromstring(str_encode, np.int32, count=1)
        print('angle:{}'.format(angle))
        data = np.fromstring(str_encode, np.uint8)
        data = data[4:]

    img = cv2.imdecode(data, cv2.IMREAD_COLOR) #by default, it's a three channel images
    img = cv2.resize(img, (FLAGS.image_size, FLAGS.image_size))
    #cv2.imwrite('tmp1.jpg', img)
    if angle == 180:
        img = np.rot90(img, 1)
    elif angle == 90:
        img = np.rot90(img, 2)
    elif angle == 0:
        img = np.rot90(img, -1)
    #cv2.imwrite('tmp2.jpg', img)

    img = img[:, :, ::-1] # BGR to RGB
    img = prewhiten(img)
    out_img = []
    out_img.append(img)
    out_img = np.float32(out_img)
    return out_img

def insecure_channel(host, port):
        channel = grpc.insecure_channel(
            target=host if port is None else '%s:%d' % (host, port),
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1)])
        return grpc.beta.implementations.Channel(channel)

html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>图片上传</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=上传>
    </form>
    '''


def inference_pipe(f_handler, local_flag):
    def inference_init():
        #print('test here, x_data shape:{}'.format(np.shape(x_data)))
        host, port = FLAGS.server.split(':')
        channel = insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
          
        # Send request
        req = predict_pb2.PredictRequest()
        req.model_spec.name = 'mobilenet_v1'
        req.model_spec.signature_name = 'predict_images'
        return stub, req

    #global variable, to initiate the tf serving pipeline config only once!!!!!!!!!!
    global global_init_flag, stub, req

    if global_init_flag == 'True':
        stub, req = inference_init()
        global_init_flag = 'False'
    print("----------------------INFERENCE STARTING ---------------------")
    StartTime = time.time()
    x_data = preprocess(f_handler, local_flag)
    print('test here, x_data:{}'.format(x_data))
    EndTime = time.time()
    print('preprocess time:{}'.format(EndTime-StartTime))
    
    startTime = time.time()
    req.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(x_data, shape=[1, FLAGS.image_size, FLAGS.image_size, 3]))
    result = stub.Predict(req, 10.0)  # 10 secs timeout
    endTime = time.time()
    #print("the first image, start time:{}, end time:{}, elapsed time:{} ", (startTime, endTime, endTime - startTime))
    """
    startTime = time.time()
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(x_data, shape=[1, FLAGS.image_size, FLAGS.image_size, 3]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    endTime = time.time()
    print("the second image, start time:{}, end time:{}, elapsed time:{} ", (startTime, endTime, endTime - startTime))
    """
    #print("result:{}\n, type:{} ".format(result, type(result)))
    print("\n\n\n\n---------------------- SUCCESSFUL ---------------------")
    return result

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/vision/recognize', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            #result = 'hello world'
            result = inference_pipe(file, local_flag=0) #init inference pipeline
            result = result.outputs['outputs'].float_val
            result_list = []
            for i in range(16):
                result_dict = {}
                result_dict['name'] = ClassInd[i]
                result_dict['value'] = result[i]
                #result_dict[ClassInd[i]] = result[i]
                result_list.append(result_dict)
            #max_ind = np.argmax(result_list)
            #output_str = '{"predict class":"' + str(ClassInd[max_ind]) + '", "probability":"' + str(result_list[max_ind]) + '"}'
            output_str = json.dumps(result_list)

            return output_str
    return html

if __name__ == '__main__':
  with open(FLAGS.local_file_name, 'rb') as fp:
    print('FLAGS.local_file_name:{}'.format(FLAGS.local_file_name))
    res = inference_pipe(fp, local_flag=1) #init inference pipeline
    print("res:{}\n, type:{} ".format(res, type(res)))
    app.run(host=FLAGS.server_out.split(':')[0], port=int(FLAGS.server_out.split(':')[1])) 
