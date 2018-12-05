''' * Stain-Color Normalization by using Deep Convolutional GMM (DCGMM).
    * VCA group, Eindhoen University of Technology.
    * Ref: Zanjani F.G., Zinger S., Bejnordi B.E., van der Laak J. AWM, de With P. H.N., "Histopathology Stain-Color Normalization Using Deep Generative Models", (2018).'''

import tensorflow as tf
import ops as utils
from GMM_M_Step import GMM_M_Step
import numpy as np
import math


class UNET(object):
  def __init__(self, name, config, is_train):
    self.name = name
    self.is_train = is_train
    self.reuse = None
    
    with tf.variable_scope(self.name, reuse=self.reuse):
        G_W1 = utils.weight_variable([3, 3, 1, 32], name="G_W1")
        G_b1 = utils.bias_variable([32], name="G_b1")
        
        G_W2 = utils.weight_variable([3, 3, 32, 32], name="G_W2")
        G_b2 = utils.bias_variable([32], name="G_b2")
        

        G_W3 = utils.weight_variable([3, 3, 32, 64], name="G_W3")
        G_b3 = utils.bias_variable([64], name="G_b3")
        
        G_W4 = utils.weight_variable([3, 3, 64, 64], name="G_W4")
        G_b4 = utils.bias_variable([64], name="G_b4")
        

        G_W5 = utils.weight_variable([3, 3, 64, 128], name="G_W5")
        G_b5 = utils.bias_variable([128], name="G_b5")
        
        G_W6 = utils.weight_variable([3, 3, 128, 128], name="G_W6") 
        G_b6 = utils.bias_variable([128], name="G_b6")
        
        G_W7 = utils.weight_variable([1, 1, 128, 128], name="G_W7") 
        G_b7 = utils.bias_variable([128], name="G_b7")



        
        G_W8 = utils.weight_variable([3, 3, 128, 256], name="G_W8")
        G_b8 = utils.bias_variable([256], name="G_b8")

        G_W9 = utils.weight_variable([3, 3, 256, 256], name="G_W9")
        G_b9 = utils.bias_variable([256], name="G_b9")

        G_W10 = utils.weight_variable([1, 1, 256, 256], name="G_W10")
        G_b10 = utils.bias_variable([256], name="G_b10")


        G_W11 = utils.weight_variable([3, 3, 256, 256], name="G_W11")
        G_b11 = utils.bias_variable([256], name="G_b11")

        G_W12 = utils.weight_variable([3, 3, 256, 256], name="G_W12")
        G_b12 = utils.bias_variable([256], name="G_b12")

        G_W13 = utils.weight_variable([1, 1, 256, 256], name="G_W13")
        G_b13 = utils.bias_variable([256], name="G_b13")


        G_W14 = utils.weight_variable([3, 3, 512, 128], name="G_W14")
        G_b14 = utils.bias_variable([128], name="G_b14")

        G_W15 = utils.weight_variable([3, 3, 256, 64], name="G_W15")
        G_b15 = utils.bias_variable([64], name="G_b15")

        G_W16 = utils.weight_variable([3, 3, 128, 32], name="G_W16")
        G_b16 = utils.bias_variable([32], name="G_b16")

        G_W17 = utils.weight_variable([3, 3, 64, 64], name="G_W17")
        G_b17 = utils.bias_variable([64], name="G_b17")

        G_W18 = utils.weight_variable([1, 1, 64, 32], name="G_W18")
        G_b18 = utils.bias_variable([32], name="G_b18")

        G_W19 = utils.weight_variable([3, 3, 32, config.ClusterNo], name="G_W19")
        G_b19 = utils.bias_variable([config.ClusterNo], name="G_b19")

        self.Param = {'G_W1':G_W1, 'G_b1':G_b1, 
                 'G_W2':G_W2, 'G_b2':G_b2,  
                 'G_W3':G_W3, 'G_b3':G_b3, 
                 'G_W4':G_W4, 'G_b4':G_b4, 
                 'G_W5':G_W5, 'G_b5':G_b5, 
                 'G_W6':G_W6, 'G_b6':G_b6, 
                 'G_W7':G_W7, 'G_b7':G_b7, 
                 'G_W8':G_W8, 'G_b8':G_b8, 
                 'G_W9':G_W9, 'G_b9':G_b9,
                 'G_W10':G_W10, 'G_b10':G_b10, 
                 'G_W11':G_W11, 'G_b11':G_b11, 
                 'G_W12':G_W12, 'G_b12':G_b12, 
                 'G_W13':G_W13, 'G_b13':G_b13,
                 'G_W14':G_W14, 'G_b14':G_b14, 
                 'G_W15':G_W15, 'G_b15':G_b15, 
                 'G_W16':G_W16, 'G_b16':G_b16, 
                 'G_W17':G_W17, 'G_b17':G_b17,
                 'G_W18':G_W18, 'G_b18':G_b18,
                 'G_W19':G_W19, 'G_b19':G_b19}
      
    if self.reuse is None:
          self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
          self.saver = tf.train.Saver(self.var_list)
          self.reuse = True         

  def __call__(self, D):
    with tf.variable_scope(self.name, reuse=self.reuse):
        
        D_norm = D 
        G_conv1 = utils.conv2d_basic(D_norm, self.Param['G_W1'], self.Param['G_b1'])
        G_relu1 = tf.nn.relu(G_conv1, name="G_relu1")
    
        G_conv2 = utils.conv2d_basic(G_relu1, self.Param['G_W2'], self.Param['G_b2'])
        G_relu2 = tf.nn.relu(G_conv2, name="G_relu2")
        
        G_pool1 = utils.max_pool_2x2(G_relu2)
        G_conv3 = utils.conv2d_basic(G_pool1, self.Param['G_W3'], self.Param['G_b3'])
        G_relu3 = tf.nn.relu(G_conv3, name="G_relu3")
        
        G_conv4 = utils.conv2d_basic(G_relu3, self.Param['G_W4'], self.Param['G_b4'])
        G_relu4 = tf.nn.relu(G_conv4, name="G_relu4")
        
        G_pool2 = utils.max_pool_2x2(G_relu4)


        
        G_conv5 = utils.conv2d_basic(G_pool2, self.Param['G_W5'], self.Param['G_b5'])
        G_relu5 = tf.nn.relu(G_conv5, name="G_relu5")
        

        G_conv6 = utils.conv2d_basic(G_relu5, self.Param['G_W6'], self.Param['G_b6'])
        G_relu6 = tf.nn.relu(G_conv6, name="G_relu6")

        G_conv7 = utils.conv2d_basic(G_relu6, self.Param['G_W7'], self.Param['G_b7'])
        G_relu7 = tf.nn.relu(G_conv7, name="G_relu7")  

        G_pool3 = utils.max_pool_2x2(G_relu7)    
        G_conv8 = utils.conv2d_basic(G_pool3, self.Param['G_W8'], self.Param['G_b8'])
        G_relu8 = tf.nn.relu(G_conv8, name="G_relu8")

        G_conv9 = utils.conv2d_basic(G_relu8, self.Param['G_W9'], self.Param['G_b9'])
        G_relu9 = tf.nn.relu(G_conv9, name="G_relu9")

        G_conv10 = utils.conv2d_basic(G_relu9, self.Param['G_W10'], self.Param['G_b10'])
        G_relu10 = tf.nn.relu(G_conv10, name="G_relu10")  

        G_pool4 = utils.max_pool_2x2(G_relu10)   
        G_conv11 = utils.conv2d_basic(G_pool4, self.Param['G_W11'], self.Param['G_b11'])
        G_relu11 = tf.nn.relu(G_conv11, name="G_relu11")

        G_conv12 = utils.conv2d_basic(G_relu11, self.Param['G_W12'], self.Param['G_b12'])
        G_relu12 = tf.nn.relu(G_conv12, name="G_relu12")

        G_conv13 = utils.conv2d_basic(G_relu12, self.Param['G_W13'], self.Param['G_b13'])
        G_relu13 = tf.nn.relu(G_conv13, name="G_relu13")  


        G_rs14 = tf.image.resize_images(G_relu13, [63, 63], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
        #G_rs14 = tf.image.resize_images(G_relu13, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
        G_cc14 = tf.concat([G_rs14, G_relu10], 3, name="G_cc14")

        G_conv14 = utils.conv2d_basic(G_cc14, self.Param['G_W14'], self.Param['G_b14'])
        G_relu14 = tf.nn.relu(G_conv14, name="G_rs14")
        
        G_rs15 = tf.image.resize_images(G_relu14, [125, 125], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  
        #G_rs15 = tf.image.resize_images(G_relu14, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)   
        G_cc15 = tf.concat([G_rs15, G_relu7], 3, name="G_cc15")

        G_conv15 = utils.conv2d_basic(G_cc15, self.Param['G_W15'], self.Param['G_b15'])
        G_relu15 = tf.nn.relu(G_conv15, name="G_rs15")

        G_rs16 = tf.image.resize_images(G_relu15, [250, 250], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
        #G_rs16 = tf.image.resize_images(G_relu15, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
        G_cc16 = tf.concat([G_rs16, G_relu4], 3, name="G_cc16")


        G_conv16 = utils.conv2d_basic(G_cc16, self.Param['G_W16'], self.Param['G_b16'])
        G_relu16 = tf.nn.relu(G_conv16, name="G_rs16")

        G_rs17 = tf.image.resize_images(G_relu16, [500, 500], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
        #G_rs17 = tf.image.resize_images(G_relu16, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
        G_cc17 = tf.concat([G_rs17, G_relu2], 3, name="G_cc17")

        G_conv17 = utils.conv2d_basic(G_cc17, self.Param['G_W17'], self.Param['G_b17'])
        G_relu17 = tf.nn.relu(G_conv17, name="G_rs17")

        G_conv18 = utils.conv2d_basic(G_relu17, self.Param['G_W18'], self.Param['G_b18'])
        G_relu18 = tf.nn.relu(G_conv18, name="G_relu18")
        
        G_conv19 = utils.conv2d_basic(G_relu18, self.Param['G_W19'], self.Param['G_b19'])
        Gama = tf.nn.softmax(G_conv19, name="G_latent_softmax")


    return Gama










  

class DCGMM(object):
  def __init__(self, sess, config, name, is_train):
    self.sess = sess
    self.name = name
    self.is_train = is_train


    self.X_hsd = tf.placeholder(tf.float32, shape=[config.batch_size, config.im_size, config.im_size, 3], name="original_color_image")
    self.D, h_s = tf.split(self.X_hsd,[1,2], axis=3)
    
    self.E_Step = UNET("E_Step", config, is_train=self.is_train)
    self.Gama = self.E_Step(self.D)

    self.loss, self.Mu, self.Std = GMM_M_Step(self.X_hsd, self.Gama, config.ClusterNo, name='GMM_Statistics')
    
    if self.is_train:

      self.optim = tf.train.AdamOptimizer(config.lr)
      self.train = self.optim.minimize(self.loss, var_list=self.E_Step.Param)

    ClsLbl = tf.arg_max(self.Gama, 3)
    ClsLbl = tf.cast(ClsLbl, tf.float32)
    
    ColorTable = [[255,0,0],[0,255,0],[0,0,255],[255,255,0], [0,255,255], [255,0,255]]
    colors = tf.cast(tf.constant(ColorTable), tf.float32)
    Msk = tf.tile(tf.expand_dims(ClsLbl, axis=3),[1,1,1,3])
    for k in range(0, config.ClusterNo):
        ClrTmpl = tf.einsum('anmd,df->anmf', tf.expand_dims(tf.ones_like(ClsLbl), axis=3), tf.reshape(colors[k,...],[1,3]))
        Msk = tf.where(tf.equal(Msk,k), ClrTmpl, Msk)
    
    
    self.X_rgb = utils.HSD2RGB(self.X_hsd)
    tf.summary.image("1.Input_image", self.X_rgb*255.0, max_outputs=2)
    tf.summary.image("2.Gamma_image",  Msk, max_outputs=2)
    tf.summary.image("3.Density_image", self.D*255.0, max_outputs=2)
    tf.summary.scalar("loss", self.loss)

    self.summary_op = tf.summary.merge_all()

    self.saver = tf.train.Saver()
    self.summary_writer = tf.summary.FileWriter(config.logs_dir, self.sess.graph)

    self.sess.run(tf.global_variables_initializer())
    
    ckpt = tf.train.get_checkpoint_state(config.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Model restored...")
   

  def fit(self, X):
    _, loss, summary_str = self.sess.run([self.train, self.loss, self.summary_op], {self.X_hsd:X})
    return loss, summary_str, self.summary_writer

  def deploy(self, X):
    mu, std, gama, summary_str = self.sess.run([self.Mu, self.Std, self.Gama, self.summary_op], {self.X_hsd:X})
    return mu, std, gama
    
  def save(self, dir_path):
    self.E_Step.save(self.sess, dir_path+"/model.ckpt")

  def restore(self, dir_path):
    self.E_Step.restore(self.sess, dir_path+"/model.ckpt")
