#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:03:58 2020

@author: bigmao
"""

import os
import scipy.io
import utils
import numpy as np
import matplotlib.pyplot as plt  


from keras.layers import LeakyReLU, Conv1D, Input, Flatten, Conv2DTranspose, Lambda, Conv2D, Reshape,BatchNormalization
from keras.layers import Dense, Activation, Dropout, Permute
from keras.optimizers import Adam, SGD

from DCGAN_model_ST_o import build_generator, build_discriminator, sigmoid_crossentropy,average_crossentropy1

from keras.models import Model, load_model
import tensorflow as tf
import random

import keras.engine
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="1"



data_folder = 'NIH_Clean_Sw'
file_ls = os.listdir(data_folder)

file_ls = [os.path.join(data_folder,item) for item in file_ls]

data_params = {'data_len':8192,'nb_channel':1,'data_size':1698,
                    'nperseg':256,'noverlap':192,'ntime':128,'nfreq':128}

real_data,smax,smin = utils.get_real_data(file_ls,data_params)

########================================================================================

data_params = {'data_len':8192,'nb_channel':1,'data_size':1698,
                    'nperseg':256,'noverlap':192,'ntime':128,'nfreq':128}
network_params = {'latent_dim':400,'dropout_rate':0,'batch_size':128,'dcgan_loss':'lsgan','use_batchnorm':False,
                    'use_MBdisc':True,'MD_nb_kernels':15,'MD_kernel_dim':40}
training_params = {'D_lr':3e-3, 'G_lr':5e-5, 'decay':0.707,'decay_epoch':1000,
                    'D_round':1, 'G_round':3, 'begin_epoch':1, 'end_epoch':2000}

########================================================================================
#D_opt = Adam(lr=training_params['D_lr'], beta_1=0.5,clipnorm=1.0)

G_lr=training_params['G_lr']
D_lr=training_params['D_lr']
decay=training_params['decay']

D_opt = SGD(lr=G_lr,clipnorm=1.0)
G_opt = Adam(lr=D_lr, beta_1=0.5,clipnorm=1.0) 


discriminator = build_discriminator(data_params,network_params)

discriminator.compile(
    optimizer=D_opt,
    loss=sigmoid_crossentropy,
)
discriminator.trainable = False


generator = build_generator(data_params,network_params)


z = Input(shape=(network_params['latent_dim'],))
generated = generator(z)



decision = discriminator(generated)  

Combined = Model(inputs=z,outputs=decision)

Combined.compile(
    optimizer=G_opt,
    loss=sigmoid_crossentropy,
)
#==========================================================================================


def th_strategy(ep):
    
    th_up = 3
    th_down = 0.8 
    ep_keep = 750 
    
    if ep<ep_keep:

        th = ((th_up-th_down)/ep_keep)*(ep_keep-ep)+th_down
        
        return th
    else:
        return th_down
        
        
        
        

results = {'ep':[],
           'g_loss':[],
           'd_loss':[],
           'd_real_loss':[],
           'd_fake_loss':[]          
        }



batch_size = network_params['batch_size']
batch_number = np.int64(np.ceil(real_data.shape[0]/batch_size))
count_record = []
smooth = 0.01
d_th = 0.80

for epoch in range(5000):
    
    
    if epoch%(training_params['decay_epoch']) == 0:
        G_lr = G_lr*decay
        D_lr = D_lr*decay
        K.set_value(Combined.optimizer.lr, G_lr)
        K.set_value(discriminator.optimizer.lr, D_lr)
    
    
    np.random.shuffle(real_data)
    ep_g_loss = []
    ep_d_loss = []
    
    ep_d_real_loss = []
    ep_d_fake_loss = []
    
    for b in range(batch_number):
        
        # prepare batch data
        batch_data = real_data[b*batch_size:(b+1)*batch_size]
        batch_real_label = np.ones((batch_data.shape[0],1))
        batch_trick = np.ones((batch_data.shape[0],1))
        batch_fake_label = np.zeros((batch_data.shape[0],1))

        
        
        
        #G step
# =============================================================================
#         for _ in range(training_params['G_round']):
#             batch_noise = np.random.uniform(-1,1,size=[batch_data.shape[0],network_params['latent_dim']])
#             Combined.train_on_batch(batch_noise,batch_trick)
# =============================================================================
        
        batch_noise = np.random.uniform(-1,1,size=[batch_data.shape[0],network_params['latent_dim']])
        g_b_loss = Combined.train_on_batch(batch_noise,batch_trick)
        #print('g_b_loss----------------------------',g_b_loss)
# =============================================================================
#         count = 1
#         while g_b_loss>1.0:
#             batch_noise = np.random.uniform(-1,1,size=[batch_data.shape[0],network_params['latent_dim']])
#             g_b_loss = Combined.train_on_batch(batch_noise,batch_trick)
#             count += 1
#         count_record.append(count)
# =============================================================================
        

        
        #g_b_loss = Combined.evaluate(batch_noise,batch_trick,verbose=0)
# =============================================================================
#         print('g_b_loss----------------------------',g_b_loss)
# =============================================================================
        
        # D step
        batch_noise = np.random.uniform(-1,1,size=[batch_data.shape[0],network_params['latent_dim']])
        batch_generated = generator.predict(batch_noise)
        
# =============================================================================
#         batch_c = np.concatenate([batch_generated,batch_data],0)        
#         label_c = np.concatenate([batch_fake_label,batch_real_label * (1 - smooth)],0)
# =============================================================================
        d_b_loss_real = discriminator.train_on_batch(batch_data,batch_real_label* (1 - smooth))
        d_b_loss_fake = discriminator.train_on_batch(batch_generated,batch_fake_label)
        
        
# =============================================================================
#         d_b_loss_fake = discriminator.evaluate(batch_generated,batch_fake_label,verbose=0)
#         d_b_loss_real = discriminator.evaluate(batch_data,batch_real_label,verbose=0)
# =============================================================================
        

        
        
        

# =============================================================================
#         print('d_b_fake+++++',d_b_loss_fake)      
#         print('d_b_real+++++',d_b_loss_real)
# =============================================================================
        
        #d_b_loss_real = discriminator.evaluate(batch_data,batch_real_label,verbose=0)
        d_b_loss = (d_b_loss_fake+d_b_loss_real)/2
        while d_b_loss_fake>d_th or d_b_loss_real>d_th:
            
            if d_b_loss_real >d_th:
                d_b_loss_real = discriminator.train_on_batch(batch_data,batch_real_label* (1 - smooth))
                #print('d_b_real============',d_b_loss_real)
            
            if d_b_loss_fake >d_th:
                batch_noise = np.random.uniform(-1,1,size=[batch_data.shape[0],network_params['latent_dim']])
                batch_generated = generator.predict(batch_noise)
                d_b_loss_fake = discriminator.train_on_batch(batch_generated,batch_fake_label)
                #print('d_b_fake+++++',d_b_loss_fake)  
            d_b_loss = (d_b_loss_fake+d_b_loss_real)/2
            
# =============================================================================
#             print('d_b_fake+++++',d_b_loss_fake)      
#             print('d_b_real+++++',d_b_loss_real)
# =============================================================================
        
        #print('----',d_b_loss)         
        
        
        ep_d_loss.append(d_b_loss)
        ep_d_real_loss.append(d_b_loss_real)
        ep_d_fake_loss.append(d_b_loss_fake)
        ep_g_loss.append(g_b_loss)
        
    
        

        
    print('now EP %d, g loss---%0.4f, d loss--%0.4f, d on real loss---%0.4f, d on fake loss--%0.4f'%(epoch,np.mean(ep_g_loss),np.mean(ep_d_loss),np.mean(ep_d_real_loss),np.mean(ep_d_fake_loss)))
    print(np.mean(count_record))
    
    plt.imshow(batch_generated[10,:,:,0]),plt.show()
    plt.imshow(batch_generated[20,:,:,0]),plt.show()
    
    results['ep'].append(epoch)
    results['g_loss'].append(np.mean(ep_g_loss))
    results['d_loss'].append(np.mean(ep_d_loss))
    results['d_real_loss'].append(np.mean(ep_d_real_loss))
    results['d_fake_loss'].append(np.mean(ep_d_fake_loss))
    
    if epoch%10==0:
        plt.plot(results['g_loss'])
        plt.plot(results['d_loss'])
        plt.plot(results['d_real_loss'])
        plt.plot(results['d_fake_loss'])
        
        plt.legend(['g_loss','d_loss','d_real_loss','d_fake_loss'])
        plt.xlabel('epoch')
        plt.show()
