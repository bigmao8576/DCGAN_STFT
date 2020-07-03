import numpy as np
import random
import scipy.io

"""Create a training set of sine waves with 10000 records"""
a = np.arange(0.1,0.9,0.02) # amplitudes between [0.1,0.9]
x = np.arange(0,256*32,0.5) # time steps
r = np.arange(2,6.1,0.1) # frequency between [2.0,6.0] rad? should be Hz
count = 0
fs = len(x)
y = np.zeros((1,len(x)))

for n in range(2000):
  amp = a[random.randint(0,len(a)-1)]
  rad = r[random.randint(0,len(r)-1)]
  phase = random.uniform(-1,1)*np.pi
  y = np.append(y,amp*np.sin(((2*np.pi*rad*x)+phase)/fs).reshape((1,len(x))),axis = 0)

file_name = 'Sine_data_int.mat'
scipy.io.savemat(file_name,{'samples':y[1:][:]})

a = np.arange(0.1,0.9,0.02)
x = np.arange(0,256*32,0.5)
r = np.arange(2,6.1,0.1)
count = 0
fs = len(x)
y = np.zeros((1,len(x)))

for n in range(200):
  amp = a[random.randint(0,len(a)-1)]
  rad = r[random.randint(0,len(r)-1)]
  phase = random.uniform(-1,1)*np.pi

  y = np.append(y,amp*np.sin(((2*np.pi*rad*x+phase))/fs).reshape((1,len(x))),axis = 0)
  
file_name = 'Sine_data_int_test.mat'
scipy.io.savemat(file_name,{'samples':y[1:][:]})