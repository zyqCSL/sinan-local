import glob
import sys
import time
import json
import logging

sys.path.append('../../gen-py')
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from social_network import MediaFilterService

#---- image processing & ml --------#
import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
# import StringIO
import io

ModelPath = "./nsfw_mobilenet2.224x224.h5"
ImageSize = (224, 224)
Categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
NSFW_Model = keras.models.load_model(ModelPath)
NSFW_Model._make_predict_function()
Graph = tf.get_default_graph()
print("nsfw model loaded")

class MediaFilterServiceHandler:
    def __init__(self):
        pass
        # global ModelPath
        # self.nsfw_model = keras.models.load_model(ModelPath)
        # self.nsfw_model._make_predict_function()
        # print("nsfw model loaded")

    def _load_base64_image(self, base64_str, image_size):
        # base64_str += "=" * ((4 - len(base64_str) % 4) % 4)  # restore stripped '='s
        try:
            img_str = base64.b64decode(base64_str)
        except Exception as e:
            print(e)
            print("faulty b64_img:")
            print([base64_str])
            sys.exit()
        # tempBuff = StringIO.StringIO()
        tempBuff = io.BytesIO()
        tempBuff.write(img_str)
        tempBuff.flush()
        # tempBuff.seek(0) #need to jump back to the beginning before handing it off to PIL
        image = Image.open(tempBuff)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(image_size)
        tempBuff.close()

        # logging.error("image data")        
        # logging.error(type(image))
        # logging.error(image.format)
        # logging.error(image.mode)
        # logging.error(image.size)

        image = keras.preprocessing.image.img_to_array(image)

        # logging.error("array data")
        # logging.error(type(image))
        # logging.error(image.shape)
        # logging.error(' ')


        image /= 255
        return image

    def _predict(self, base64_images, image_size):
        global NSFW_Model
        global Categories
        global Graph
        images = []
        for img in base64_images:
            images.append(self._load_base64_image(img, image_size))
        images = np.asarray(images)

        _return = []
        try:
            with Graph.as_default():
                model_preds = NSFW_Model.predict(images, batch_size = len(images))
                preds = np.argsort(model_preds, axis = 1).tolist()

            for i, single_preds in enumerate(preds):
                _type = Categories[single_preds[-1]]
                print(_type)
                _flag = (_type != "porn" and _type != "hentai")
                _return.append(_flag)

        except Exception as e:
            print("prediction failed")
            print(e)
            for i in range(0, len(base64_images)):
                _return.append(False)

        return _return

    def UploadMedia(self, req_id, media_types, medium, carrier):
        global ImageSize
        if len(medium) == 0:
            return []
        # print(media_types)
        # print(medium)
        start = time.time()
        _return = self._predict(medium, ImageSize)
        # _return = []
        # try:
        #     _return = self._predict(base64_images, ImageSize)
        # except:
        #     print("Error when predicting")
        #     for i in range(0, len(medium)):
        #         _return.append(False)
        end = time.time()
        print("inference time = %.2fs" %(end - start))
        print(_return)
        return _return

# -------------------- Thrift server impl ----------------------#
if __name__ == '__main__':
    host_addr = 'localhost'
    host_port = 40000
    with open('../../config/service-config.json', 'r') as f:
        config_json_data = json.load(f)
        host_addr = config_json_data['media-filter-service']['addr']
        host_port = int(config_json_data['media-filter-service']['port'])

    print host_addr, ' ', host_port
    handler = MediaFilterServiceHandler()
    processor = MediaFilterService.Processor(handler)
    transport = TSocket.TServerSocket(host=host_addr, port=host_port)
    tfactory = TTransport.TFramedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    # server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    
    server = TServer.TThreadedServer(
        processor, transport, tfactory, pfactory)
    # server = TServer.TThreadPoolServer(
    #     processor, transport, tfactory, pfactory)

    print('Starting MediaFilterService Server...')
    server.serve()
    print('done.')