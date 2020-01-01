
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal
from matplotlib import pyplot as plt

from signal_python.world import main
from signal_python.world import synthesisRequiem
from signal_python.world.get_seeds_signals import get_seeds_signals
from signal_python.lpc import lpc
from signal_python.glottal import gmf_iaif
import pysptk
import librosa
import time
import sys
from signal_python.world import cheaptrick
from scipy.signal import lfilter
from scipy.signal import hanning
from scipy.fftpack import fft, ifft
from scipy.signal import fftconvolve

wav_path = sys.argv[1]

x,fs = librosa.load(wav_path,sr=24000)

vocoder = main.World()

# analysis
start_time=time.time()
dat,_ = vocoder.encode(fs, x, f0_method='swipe', is_requiem=True) # use requiem analysis and synthesis
print("word analysis cost {}".format(time.time()-start_time))

ORDER=24

def psd2lpc(psd):

    glottal_lpcs=[]
    glottal_gains=[]
    lpcs=[]
    gains=[]
    
    start_time = time.time()
    for i in range(psd.shape[1]):
        lpc_coef,g=lpc.psd2lpc(psd[:,i],order=ORDER)
        glottal_lpc,glottal_g,lpc_coef,g=gmf_iaif.gmf_iaif(psd[:,i],vt_order=ORDER)

        glottal_lpcs.append(glottal_lpc)
        glottal_gains.append(glottal_g)
        lpcs.append(lpc_coef)
        gains.append(g)
    print("gmf-iaif cost :{}, frames: {}".format(time.time()-start_time,psd.shape[1]))

    return np.array(glottal_lpcs),np.array(glottal_gains),np.array(lpcs),np.array(gains)




def inverse_lpc_fftconvolve(x,dat):
    start_time = time.time()
    fft_size = (dat['spectrogram'].shape[0]-1)*2
    glottal_lpcs,glottal_gains,lpcs,gains=psd2lpc(dat['spectrogram'])
    print("psd2lpc cost {}".format(time.time()-start_time))
    x_res = np.zeros([x.shape[0]+100])
    x_glottal_res = np.zeros([x.shape[0]+100])
    f0_sequence = dat['f0']
    temporal_positions = dat['temporal_positions']

    start_time=time.time()
    recons_psds=[]
    recons_vt_psds=[]
    for glottal_lpc,glottal_g,lpc_coef,g in zip(glottal_lpcs,glottal_gains,lpcs,gains):
        recons_psd = lpc.lpc2psd(glottal_lpc,glottal_g,fft_size)
        recons_vt_psd = lpc.lpc2psd(lpc_coef,g,fft_size)
        recons_psd *=  recons_vt_psd
        recons_psds.append(recons_psd)
        recons_vt_psds.append(recons_vt_psd)
    recons_psds = np.array(recons_psds)
    recons_vt_psds = np.array(recons_vt_psds)
    print("construce psd cost {}".format(time.time()-start_time))

    start_time = time.time()
    for i in range(2,len(f0_sequence)-1):
        f0 = f0_sequence[i]
        temporal_position=temporal_positions[i]
        half_win_length = int(0.005*fs)
        win = hanning(2*half_win_length)

        base_index=np.arange(-half_win_length,half_win_length)
        index = int(temporal_position*fs+0.501) + 1.0+  base_index

        safe_index = np.minimum(len(x)-1,np.maximum(1,cheaptrick.round_matlab(index)))
        safe_index = np.array(safe_index,dtype=np.int)

        win_x = x[safe_index] * win


        """conv implementation"""
        inv = fftconvolve(win_x,lpcs[i,:]/gains[i],mode="same") 
        x_glottal_res[safe_index] += inv
        inv = fftconvolve(inv,glottal_lpcs[i,:]/glottal_gains[i],mode="same") 
        x_res[safe_index] += inv 

    print("filter cost {}".format(time.time()-start_time))
    return x_res, x_glottal_res,recons_psds,recons_vt_psds




x_res,x_glottal_res,recons_psds,recons_vt_psds = inverse_lpc_fftconvolve(x,dat)

wavwrite('x_res.wav', fs, (x_res * 2 ** 15).astype(np.int16))
wavwrite('x_glottal_res.wav', fs, (x_glottal_res * 2 ** 15).astype(np.int16))

shift = -12
import pyrubberband as pyrb
x_res = pyrb.pitch_shift(x_res,fs,shift)

y = synthesisRequiem.get_waveform(x_res,
                    np.transpose(recons_psds,[1,0]),
                    dat['temporal_positions'],
                    dat['f0'],
                    dat['fs'])

x_glottal_res=pyrb.pitch_shift(x_glottal_res,fs,shift)
y_from_glottal = synthesisRequiem.get_waveform(x_glottal_res,
                    np.transpose(recons_vt_psds,[1,0]),
                    dat['temporal_positions'],
                    dat['f0'],
                    dat['fs'])

wavwrite('x_recons.wav', fs, (y * 2 ** 15).astype(np.int16))
wavwrite('x_recons_glottal.wav', fs, (y_from_glottal * 2 ** 15).astype(np.int16))

from scipy import interpolate
import copy

import time

def semitone2ratio(semitone):
    return np.power(10,semitone*100*np.log10(2)/1200)

ratio = semitone2ratio(shift)

fft_size = (dat['spectrogram'].shape[0]-1)*2

start_time = time.time()
dat_shift = copy.deepcopy(dat) 
plt.subplot(2,1,1)
plt.plot(dat_shift['spectrogram'][:,300])  
print(dat_shift['spectrogram'][-20:,300])
for i in range(dat_shift['spectrogram'].shape[1]):
    interp_f = interpolate.interp1d(np.arange(0,fft_size//2+1),dat_shift['spectrogram'][:,i],kind='cubic',bounds_error=False,fill_value='extrapolate')
    dat_shift['spectrogram'][:,i] = interp_f(np.arange(0,fft_size//2+1)*ratio)
    #dat_shift['spectrogram'][:,i] = np.interp(dat_shift['spectrogram'][:,i],np.arange(0,fft_size//2+1),np.arange(0,fft_size//2+1)*ratio)
plt.subplot(2,1,2)
plt.plot(dat_shift['spectrogram'][:,300])  
print(dat_shift['spectrogram'][-20:,300])
plt.savefig("test.png")
print("interp time: {}".format(time.time()-start_time))
x_res,x_glottal_res,recons_psds,recons_vt_psds = inverse_lpc_fftconvolve(x,dat)
y = synthesisRequiem.get_waveform(x_res,
                    dat_shift['spectrogram'],
                    dat_shift['temporal_positions'],
                    dat_shift['f0'],
                    dat_shift['fs'])
wavwrite('x_prwrap.wav', fs, (y * 2 ** 15).astype(np.int16))
x_shift = pyrb.pitch_shift(y,fs,shift)
wavwrite('x_prwap_shift.wav', fs, (x_shift * 2 ** 15).astype(np.int16))






seeds_signals = get_seeds_signals(fs)
dat,_ = vocoder.encode(fs, x, f0_method='swipe', is_requiem=True) # use requiem analysis and synthesis
dat['f0'] = dat['f0']*ratio

y = synthesisRequiem.synthesisRequiem(dat,dat,seeds_signals)

wavwrite('x_shift_world.wav', fs, (y * 2 ** 15).astype(np.int16))


