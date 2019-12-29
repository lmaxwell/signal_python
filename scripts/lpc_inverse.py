from pathlib import Path

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal

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

wav_path = sys.argv[1]

x,fs = librosa.load(wav_path,sr=24000)

vocoder = main.World()

# analysis
start_time=time.time()
dat,_ = vocoder.encode(fs, x, f0_method='swipe', is_requiem=True) # use requiem analysis and synthesis
print("word analysis cost {}".format(time.time()-start_time))

ORDER=48

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

def calculate_windowed_waveform(x: np.ndarray, fs: int, f0: float, temporal_position: float) -> np.ndarray:
    '''
    First step: F0-adaptive windowing
    Design a window function with basic idea of pitch-synchronous analysis.
    A Hanning window with length 3*T0 is used.
    Using the window makes over all power of the periodic signal temporally stable

    '''
    half_window_length = int(1.5 * fs / f0 + 0.5)
    base_index = np.arange(-half_window_length, half_window_length + 1)
    index = int(temporal_position * fs + 0.501) + 1.0 + base_index
    safe_index = np.minimum(len(x), np.maximum(1, cheaptrick.round_matlab(index)))
    safe_index = np.array(safe_index, dtype=np.int)
    
    #  wave segments and set of windows preparation
    segment = x[safe_index - 1]
    time_axis = base_index / fs / 1.5
    window = 0.5 * np.cos(np.pi * time_axis * f0) + 0.5
    window /= np.sqrt(np.sum(window ** 2))
    waveform = segment * window 
    return waveform

glottal_lpcs,glottal_gains,lpcs,gains=psd2lpc(dat['spectrogram'])

#glottal_lpcs[:,0] = np.log(glottal_gains)
#lpcs[:,0] = np.log(gains)
gains = np.expand_dims(gains,axis=1)
#lpcs = np.divide(lpcs,gains)


fft_size = (dat['spectrogram'].shape[0]-1)*2

f0_low_limit = 71
default_f0 = 500
fft_size = int(2 ** np.ceil(np.log2(3 * fs / f0_low_limit + 1)))

f0_low_limit = fs * 3.0 / (fft_size - 3.0)
temporal_positions = dat['temporal_positions']
f0_sequence = dat['f0']
f0_sequence[dat['vuv'] == 0] = default_f0

spectrogram = np.zeros([int(fft_size // 2) + 1, len(f0_sequence)])
pitch_syn_spectrogram = 1j * np.zeros([int(fft_size), len(f0_sequence)])
for i in range(len(f0_sequence)):
    if f0_sequence[i] < f0_low_limit:
        f0_sequence[i] = default_f0

x_res = np.zeros([x.shape[0]+100])
x_glottal_res = np.zeros([x.shape[0]+100])

for i in range(len(f0_sequence)):
    f0 = f0_sequence[i]
    temporal_position=temporal_positions[i]
    #half_win_length = int(0.015*fs)
    half_win_length = int(1.5 * fs / f0 + 0.5)
    win = hanning(2*half_win_length)

    base_index=np.arange(-half_win_length,half_win_length)
    index = int(temporal_position*fs+0.501) + 1.0+  base_index

    safe_index = np.minimum(len(x)-1,np.maximum(1,cheaptrick.round_matlab(index)))
    safe_index = np.array(safe_index,dtype=np.int)
    win_x = x[safe_index] * win
    inv = lfilter(lpcs[i,:]/gains[i],1,win_x)
    x_glottal_res[safe_index] += inv 
    inv = lfilter(glottal_lpcs[i,:]/glottal_gains[i],1,inv)
    x_res[safe_index] += inv 






wavwrite('x_res.wav', fs, (x_res * 2 ** 15).astype(np.int16))
wavwrite('x_glottal_res.wav', fs, (x_glottal_res * 2 ** 15).astype(np.int16))


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


y = synthesisRequiem.get_waveform(x_res,
                    np.transpose(recons_psds,[1,0]),
                    dat['temporal_positions'],
                    dat['f0'],
                    dat['fs'])

y_from_glottal = synthesisRequiem.get_waveform(x_glottal_res,
                    np.transpose(recons_vt_psds,[1,0]),
                    dat['temporal_positions'],
                    dat['f0'],
                    dat['fs'])

wavwrite('x_recons.wav', fs, (y * 2 ** 15).astype(np.int16))
wavwrite('x_recons_glottal.wav', fs, (y_from_glottal * 2 ** 15).astype(np.int16))


