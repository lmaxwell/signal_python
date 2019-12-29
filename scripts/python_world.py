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

wav_path = sys.argv[1]

x,fs = librosa.load(wav_path,sr=24000)

vocoder = main.World()

# analysis
start_time=time.time()
dat,_ = vocoder.encode(fs, x, f0_method='swipe', is_requiem=True) # use requiem analysis and synthesis
print("word analysis cost {}".format(time.time()-start_time))
if 0:  # global pitch scaling
    dat = vocoder.scale_pitch(dat, 1.5)
if 0:  # global duration scaling
    dat = vocoder.scale_duration(dat, 2)
if 0:  # fine-grained duration modification
    vocoder.modify_duration(dat, [1, 1.5], [0, 1, 3, -1])  # TODO: look into this

ORDER=24

def synthesisLPC2PSD(source_object,filter_object,seeds_signals):
    fft_size = (filter_object["spectrogram"].shape[0]-1)*2
    print("synthesis lpc 2 psd")

    excitation_signal,_,_ = synthesisRequiem.get_excitation_signal(source_object['temporal_positions'],
                                              filter_object['fs'],
                                              source_object['f0'],
                                              source_object['vuv'],
                                              seeds_signals['pulse'],
                                              seeds_signals['noise'],
                                              source_object['aperiodicity'])

    psd = filter_object['spectrogram']

    lpcs=[]
    gains=[]

    start_time = time.time()
    
    for i in range(psd.shape[1]):
        lpc_coef,g=lpc.psd2lpc(psd[:,i],order=ORDER)
        lpcs.append(lpc_coef)
        gains.append(g)
    print("lpc coef cost {}".format(time.time()-start_time))
    recons_psds = []

    start_time = time.time()
    for lpc_coef,g in zip(lpcs,gains):
        recons_psd = lpc.lpc2psd(lpc_coef,g,fft_size)
        recons_psds.append(recons_psd)
    recons_psds=np.array(recons_psds)
    y0 = synthesisRequiem.get_waveform(excitation_signal,
                     np.transpose(recons_psds,[1,0]),
                     source_object['temporal_positions'],
                     source_object['f0'],
                     filter_object['fs'])
    print("filter cost {}".format(time.time() - start_time))
    return y0
    

def synthesisPSD(source_object,filter_object,seeds_signals):
    fft_size = (filter_object["spectrogram"].shape[0]-1)*2
    print("synthesis psd")
    excitation_signal,_,_ = synthesisRequiem.get_excitation_signal(source_object['temporal_positions'],
                                              filter_object['fs'],
                                              source_object['f0'],
                                              source_object['vuv'],
                                              seeds_signals['pulse'],
                                              seeds_signals['noise'],
                                              source_object['aperiodicity'])


    start_time=time.time()
    y = synthesisRequiem.get_waveform(excitation_signal,
                     filter_object['spectrogram'],
                     source_object['temporal_positions'],
                     source_object['f0'],
                     filter_object['fs'])
    print("filter cost {}".format(time.time() - start_time))
    return y

def synthesisLPC(source_object, filter_object, seeds_signals):
    fft_size = (filter_object["spectrogram"].shape[0]-1)*2
    print("synthesis lpc")
    excitation_signal,_,_ = synthesisRequiem.get_excitation_signal(source_object['temporal_positions'],
                                              filter_object['fs'],
                                              source_object['f0'],
                                              source_object['vuv'],
                                              seeds_signals['pulse'],
                                              seeds_signals['noise'],
                                              source_object['aperiodicity'])


    psd = filter_object['spectrogram']
    lpcs=[]
    gains=[]
    
    start_time=time.time()
    for i in range(psd.shape[1]):
        lpc_coef,g=lpc.psd2lpc(psd[:,i],order=ORDER)
        lpcs.append(lpc_coef)
        gains.append(g)
    print("lpc coef cost {}".format(time.time()-start_time))

    lpcs = np.array(lpcs)
    gains = np.array(gains)
    lpcs[:,0]=np.log(gains)

    poledf=pysptk.synthesis.AllPoleDF(ORDER)

    synthesizer = pysptk.synthesis.Synthesizer(poledf,int(filter_object['fs']*0.005))

    start_time=time.time()
    y=synthesizer.synthesis(excitation_signal,lpcs)
    print("filter cost {}".format(time.time() - start_time))
    return y

def synthesisGMF_IAIF(source_object,filter_object,seeds_signals):
    fft_size = (filter_object["spectrogram"].shape[0]-1)*2
    print("synthesis gmf-iaif")

    excitation_signal,_,_ = synthesisRequiem.get_excitation_signal(source_object['temporal_positions'],
                                              filter_object['fs'],
                                              source_object['f0'],
                                              source_object['vuv'],
                                              seeds_signals['pulse'],
                                              seeds_signals['noise'],
                                              source_object['aperiodicity'])

    psd = filter_object['spectrogram']


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




    recons_psds = []
    start_time = time.time()
    for glottal_lpc,glottal_g,lpc_coef,g in zip(glottal_lpcs,glottal_gains,lpcs,gains):
        recons_psd = lpc.lpc2psd(glottal_lpc,glottal_g*g,fft_size)
        #recons_psd *= lpc.lpc2psd(lpc_coef,g,fft_size)
        recons_psds.append(recons_psd)
    recons_psds=np.array(recons_psds)

    y0 = synthesisRequiem.get_waveform(excitation_signal,
                     np.transpose(recons_psds,[1,0]),
                     source_object['temporal_positions'],
                     source_object['f0'],
                     filter_object['fs'])
    print("filter cost {}".format(time.time() - start_time))
    return y0

seeds=get_seeds_signals(fs)
y0 = synthesisLPC(dat,dat,seeds)
y1 = synthesisLPC2PSD(dat,dat,seeds)
y2 = synthesisPSD(dat,dat,seeds)
y3 = synthesisGMF_IAIF(dat,dat,seeds)



# dat['f0'] = np.r_[np.zeros(5), dat['f0'][:-5]]

# synthesis
#dat = vocoder.decode(dat)
if 0:  # audio
    import simpleaudio as sa
    snd = sa.play_buffer((dat['out'] * 2 ** 15).astype(np.int16), 1, 2, fs)
    snd.wait_done()
if 0:  # visualize
    vocoder.draw(x, dat)

wavwrite('resynth0.wav', fs, (y0 * 2 ** 15).astype(np.int16))
wavwrite('resynth1.wav', fs, (y1 * 2 ** 15).astype(np.int16))
wavwrite('resynth2.wav', fs, (y2 * 2 ** 15).astype(np.int16))
wavwrite('resynth3.wav', fs, (y3 * 2 ** 15).astype(np.int16))