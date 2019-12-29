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


glottal_lpcs,glottal_gains,lpcs,gains=psd2lpc(dat['spectrogram'])

gains = np.expand_dims(gains,axis=1)
glottal_gains = np.expand_dims(glottal_gains,axis=1)

fft_size = (dat['spectrogram'].shape[0]-1)*2
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

df = pysptk.synthesis.AllZeroDF(ORDER)
synthesizer = pysptk.synthesis.Synthesizer(df,int(fs*0.005))
x_glottal_res_zerodf=synthesizer.synthesis(x,lpcs/gains)
df = pysptk.synthesis.AllZeroDF(3)
synthesizer = pysptk.synthesis.Synthesizer(df,int(fs*0.005))
x_res_zerodf=synthesizer.synthesis(x_glottal_res_zerodf,glottal_lpcs/glottal_gains)

wavwrite('x_glottal_res_zerodf.wav', fs, (x_glottal_res_zerodf * 2 ** 15).astype(np.int16))
wavwrite('x_res_zerodf.wav', fs, (x_res_zerodf * 2 ** 15).astype(np.int16))

y = synthesisRequiem.get_waveform(x_res_zerodf,
                    np.transpose(recons_psds,[1,0]),
                    dat['temporal_positions'],
                    dat['f0'],
                    dat['fs'])

y_from_glottal = synthesisRequiem.get_waveform(x_glottal_res_zerodf,
                    np.transpose(recons_vt_psds,[1,0]),
                    dat['temporal_positions'],
                    dat['f0'],
                    dat['fs'])

wavwrite('x_recons_zerodf.wav', fs, (y * 2 ** 15).astype(np.int16))
wavwrite('x_recons_glottal_zerodf.wav', fs, (y_from_glottal * 2 ** 15).astype(np.int16))







x_res = np.zeros([x.shape[0]+100])
x_glottal_res = np.zeros([x.shape[0]+100])

temporal_positions = dat['temporal_positions']
f0_sequence = dat['f0']
for i in range(len(f0_sequence)):
    f0 = f0_sequence[i]
    temporal_position=temporal_positions[i]
    half_win_length = int(0.005*fs)
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


