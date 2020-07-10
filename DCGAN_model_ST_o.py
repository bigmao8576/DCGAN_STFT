import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LeakyReLU, Conv1D, Input, Flatten, Conv2DTranspose, Lambda, Conv2D, Reshape,BatchNormalization
from keras.layers import Dense, Activation, Dropout, Reshape, Permute
from keras.layers import GRU, LSTM
from keras.optimizers import Adam
from keras import backend as K
from discrimination import MinibatchDiscrimination
from keras.constraints import Constraint

def sigmoid_crossentropy(y_true, y_pred):
    loss = K.mean(K.maximum(y_pred,0) - y_true*y_pred + K.log(1+K.exp(-K.abs(y_pred))))
    return loss

def average_crossentropy1(y_true, y_pred):
    #loss = -K.mean(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred),axis=-1)
    #loss = K.mean(K.maximum(y_pred,0) - y_true*y_pred + K.log(1+K.exp(-K.abs(y_pred))),axis=-1)
    ex = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred, name=None)
    loss = K.mean(ex,axis=1)
    return loss

def build_discriminator(data_params,network_params):
    ####input_signal = Input(shape=(data_params['ntime'],data_params['nfreq'],1))
    input_signal = Input(shape=(data_params['ntime'],data_params['nfreq'],2))

    # 1st layer [128,128,2]->[64,64,64]
    fake = Conv2D(64,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last', use_bias=False)(input_signal)
    fake = BatchNormalization()(fake,training=True)
    fake = LeakyReLU(alpha=0.2)(fake)
    # 2nd layer [64,64,64]->[32,32,128]
    fake = Conv2D(128,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last', use_bias=False)(fake)
    fake = BatchNormalization()(fake,training=True)
    fake = LeakyReLU(alpha=0.2)(fake)
    # 3rd layer [32,32,128]->[16,16,256]
    fake = Conv2D(256,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last', use_bias=False)(fake)
    fake = BatchNormalization()(fake,training=True)
    fake = LeakyReLU(alpha=0.2)(fake)
    # 4th layer [16,16,256]->[8,8,512]
    fake = Conv2D(512,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last', use_bias=False)(fake)
    fake = BatchNormalization()(fake,training=True)
    fake = LeakyReLU(alpha=0.2)(fake)
    # 5th layer [8,8,512]->[4,4,1024]
    fake = Conv2D(1024,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last', use_bias=False)(fake)
    fake = BatchNormalization()(fake,training=True)
    fake = LeakyReLU(alpha=0.2)(fake)
    fake = Flatten()(fake)
    
    
    fake = Dense(1,activation=None)(fake)
    Dis = Model(inputs=input_signal,outputs=fake)
    return (Dis)

def build_generator(data_params,network_params):
    input_noise = Input(shape=(network_params['latent_dim'],))
    # FC and reshape [100,]->[4,4,1024]
    fake_signal = Dense(256*64,activation=None,use_bias=False)(input_noise)
    fake_signal = Reshape((-1,4,1024))(fake_signal)
    fake_signal = LeakyReLU(alpha=0.2)(fake_signal)
    # 1st layer [4,4,1024]->[8,8,512]
    fake_signal = Conv2DTranspose(512,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = BatchNormalization()(fake_signal,training=True)
    fake_signal = LeakyReLU(alpha=0.2)(fake_signal)
    # 2nd layer [8,8,512]->[16,16,256]
    fake_signal = Conv2DTranspose(256,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = BatchNormalization()(fake_signal,training=True)
    fake_signal = LeakyReLU(alpha=0.2)(fake_signal)
    # 3rd layer [16,16,256]->[32,32,128]
    fake_signal = Conv2DTranspose(128,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = BatchNormalization()(fake_signal,training=True)
    fake_signal = LeakyReLU(alpha=0.2)(fake_signal)
    # 4th layer [32,32,128]->[64,64,64]
    fake_signal = Conv2DTranspose(64,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = BatchNormalization()(fake_signal,training=True)
    fake_signal = LeakyReLU(alpha=0.2)(fake_signal)
    # 5th layer [64,64,64]->[128,128,2]
    #fake_signal = Conv2DTranspose(1,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = Conv2DTranspose(2,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',activation='tanh', use_bias=False)(fake_signal)
    
    Gen = Model(inputs=input_noise,outputs=fake_signal)
    return (Gen)



#Single connected layer: https://stackoverflow.com/questions/56825036/make-a-non-fully-connected-singly-connected-neural-network-in-keras
        
