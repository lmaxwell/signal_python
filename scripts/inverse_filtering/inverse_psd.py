
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
from scipy.fftpack import fft, ifft
from scipy.signal import fftconvolve
import os

wav_path,out_path = sys.argv[1:]

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

def inverse_psd(x,dat):
    temporal_positions = dat['temporal_positions']
    f0_sequence = dat['f0']
    x_res = np.zeros([x.shape[0]+100])
    fft_size = (dat['spectrogram'].shape[0]-1)*2
    latter_index = np.arange(int(fft_size // 2 + 1), fft_size+1)
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

        spec = dat['spectrogram'][:,i]
        periodic_spectrum = np.r_[spec, spec[-2:0:-1]]
        tmp_cepstrum = np.fft.fft(np.log(np.abs(periodic_spectrum)) / 2).real
        tmp_complex_cepstrum = np.zeros(fft_size)
        tmp_complex_cepstrum[latter_index.astype(int) - 1] = tmp_cepstrum[latter_index.astype(int) - 1] * 2
        tmp_complex_cepstrum[0] = tmp_cepstrum[0]

        spectrum = np.exp(np.fft.ifft(tmp_complex_cepstrum))
        response = ifft(1/spectrum * fft(win_x, fft_size)).real
        inv = response
        origin = int(temporal_position*fs+0.501) + 1
        safe_index = np.minimum(len(x_res)-1, np.arange(origin, origin+fft_size))
        x_res[safe_index] += inv 

    print("filter cost {}".format(time.time()-start_time))
    return x_res


x_res = inverse_psd(x,dat)



os.makedirs(out_path,exist_ok=True)
wavwrite(os.path.join(out_path,'x_res.wav'), fs, (x_res * 2 ** 15).astype(np.int16))

y = synthesisRequiem.get_waveform(x_res,
                    dat['spectrogram'],
                    dat['temporal_positions'],
                    dat['f0'],
                    dat['fs'])


wavwrite(os.path.join(out_path,'x_recons.wav'), fs, (y * 2 ** 15).astype(np.int16))


