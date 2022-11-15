import tensorflow as tf
import tensorflow.keras.models as Models
import tensorflow.keras.applications.mobilenet_v2
import efficientnet.keras as efn


import PIL.Image
import numpy as np
import pickle

from src.constant import Path
from common.config import get_config


class Model:
    def __init__(self, path):         
        self.model = Models.load_model(path)
        
        self.conf = get_config()
        self.dim = self.conf['image']['dim']
        self.channel = self.conf['image']['channel']
        self.input_shape = self.conf['image']['input_shape']
        self.batch_size = self.conf['image']['batch_size']


    def preprocess(self,im):
        imgs = im.convert('RGB').resize(self.dim, resample= 0)
        
        img_arr = (np.array(imgs,dtype=np.float32))/255
        img_arr = img_arr.reshape(1, self.dim[0], self.dim[1], self.channel[0])
        return img_arr


    def predict_proba(
            self,
            img: PIL.Image.Image,
            k : int,
            show: bool = False):

        if show:
            img.show()

        im = self.preprocess(img)
        pred = self.model.predict(im)
        pred = pred.flatten()
        pred_prob = float(np.max(pred))
        # pred_prob = round(pred_prob,5)

        
        #get label
        with open(Path.LABEL, 'rb') as f:
            labels = pickle.load(f)
        
        #top
        top_k_prob, top_k_class = tf.nn.top_k(pred, k=k)
        formatted_out = []

        for pred_prob, pred_idx in zip(top_k_prob,top_k_class):
            lab = [*labels][pred_idx]
            prc_prob = float(pred_prob * 100)
            # prc_prob = round(prc_prob,6)
            formatted_out.append((lab, f"{prc_prob:.6f}   %"))

        return formatted_out
    


        
        




         

        
        

        

        
        
