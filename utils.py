import numpy as np
import matplotlib.pyplot as plt 
import keras.backend as K
import keras.layers
import os
import scipy.io
from scipy.signal import stft, istft

def sine_wave(seq_length=64, num_samples=28*5*100, num_signals=1,
        freq_low=1, freq_high=5, amplitude_low = 0.1, amplitude_high=0.9, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for _ in range(num_samples):
        signals = []
        for _ in range(num_signals):
            # f = np.random.uniform(low=freq_low, high=freq_high)     # frequency
            f = np.random.randint(low=freq_low, high=freq_high)
            A = np.random.uniform(low=amplitude_low, high=amplitude_high)        # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset)) # sampling frequency is 30Hz
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples)
    return samples
        
def draw_train_loss(plotdata,path,epoch):
    D_line, = plt.plot(plotdata['D_idx'],np.array(plotdata["D_loss"]),'b',label='Discriminator loss') 
    G_line, = plt.plot(plotdata['G_idx'],np.array(plotdata["G_loss"]),'r',label='Generator loss') 
    plt.legend(handles=[D_line,G_line])
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plot_name = 'training_loss_epoch'+str(epoch)+'.png'

    plt.savefig(os.path.join(path,plot_name))
    plt.close()

def draw_D_loss(plotdata,path,epoch):
    real_line, = plt.plot(plotdata['eval_idx'],plotdata['Dreal_loss'],'g',label='D loss in real data')
    fake_line, = plt.plot(plotdata['eval_idx'],plotdata['Dfake_loss'],'m',label='D loss in fake data')
    plt.legend(handles=[real_line,fake_line])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plot_name = 'discriminator_loss_epoch'+str(epoch)+'.png'

    plt.savefig(os.path.join(path,plot_name))
    plt.close()

def draw_generated_signal(plotdata,path,epoch):
    data1 = plotdata[0,:]#300
    data2 = plotdata[1,:]#600
    data3 = plotdata[2,:]
    data4 = plotdata[3,:]
    data5 = plotdata[4,:]

    plt.subplot(511)
    plt.plot(np.arange(len(data1)),data1)
    plt.subplot(512)
    plt.plot(np.arange(len(data2)),data2)
    plt.subplot(513)
    plt.plot(np.arange(len(data3)),data3)
    plt.subplot(514)
    plt.plot(np.arange(len(data4)),data4)
    plt.subplot(515)
    plt.plot(np.arange(len(data5)),data5)
    plt.suptitle('generated samples after '+str(epoch)+' epoch')

    plotname = 'Generated_signal_'+str(epoch)+'_epoch.png'
    plt.savefig(os.path.join(path,plotname))
    plt.close()

def draw_generated_signal_ver2(plotdata,path,epoch):
    data1 = plotdata[0,:]
    data2 = plotdata[1,:]
    data3 = plotdata[2,:]
    data4 = plotdata[3,:]
    data5 = plotdata[4,:]
    data6 = plotdata[5,:]

    plt.subplot(321)
    plt.plot(np.arange(len(data1)),data1)
    plt.subplot(322)
    plt.plot(np.arange(len(data2)),data2)
    plt.subplot(323)
    plt.plot(np.arange(len(data3)),data3)
    plt.subplot(324)
    plt.plot(np.arange(len(data4)),data4)
    plt.subplot(325)
    plt.plot(np.arange(len(data5)),data5)
    plt.subplot(326)
    plt.plot(np.arange(len(data6)),data6)
    plt.suptitle('generated samples after '+str(epoch)+' epoch')

    plotname = 'Additional generated_signal_'+str(epoch)+'_epoch.png'
    plt.savefig(os.path.join(path,plotname))
    plt.close()

def get_batch_signal(data_params,data_list,data_path,start_index,end_index):
    batch_size = end_index-start_index
    batch_signal = np.zeros((batch_size,data_params['data_len'],data_params['nb_channel']))
    for idx in range(batch_size):
        file_name = data_list[idx+start_index]
        if os.path.exists(os.path.join(data_path,file_name+'.mat')):
            signal_data = scipy.io.loadmat(os.path.join(data_path,file_name+'.mat'))
            ap_signal = signal_data[file_name][1,:]
            signal_length = len(ap_signal)
            batch_signal[idx,:signal_length,:] = np.reshape(ap_signal,(-1,1))
        else:
            raise Exception('Missing file for the batch generation')
    return batch_signal






