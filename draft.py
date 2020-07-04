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


data_folder = 'NIH_Clean_Sw'
file_ls = os.listdir(data_folder)

file_ls = [os.path.join(data_folder,item) for item in file_ls]

data_params = {'data_len':8192,'nb_channel':1,'data_size':1698,
                    'nperseg':256,'noverlap':192,'ntime':128,'nfreq':128}

real_data,smax,smin = utils.get_real_data(file_ls,data_params)





  
    
    
    
    