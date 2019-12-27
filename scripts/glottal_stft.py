import librosa 
import numpy as np
from matplotlib import pyplot as plt
import pyworld
from scipy.signal import freqz
import sys

def levinson(R,order):
    """ input: autocorrelation and order, output: LPC coefficients """
    a   = np.zeros(order+2)
    a[0] = -1
    k = np.zeros(order+1)
    # step 1: initialize prediction error "e" to R[0]
    e = R[0]
    # step 2: iterate over [1:order]
    for i in range(1,order+1):
        # step 2-1: calculate PARCOR coefficients
        k[i]= (R[i] - np.sum(a[1:i] * R[i-1:0:-1])) / e
        # step 2-2: update LPCs
        a[i] = np.copy(k[i])
        a_old = np.copy(a) 
        for j in range(1,i):
            a[j] -=  k[i]* a_old[i-j] 
        # step 2-3: update prediction error "e" 
        e = e * (1.0 - k[i]**2)
    return -1*a[0:order+1], e, -1*k[1:]

def lpc_python(psd, order):
    # N: compute next power of 2
    r = np.fft.irfft(psd)[:order+1]
    lpc,_,_ = levinson(r, order)
    g = np.sqrt(np.sum(lpc * r))
    return lpc,g
wavfile = sys.argv[1]

def main():

    fs = 16000
    x,fs=librosa.load(wavfile,fs)
    stft=librosa.core.stft(x,n_fft=1024,hop_length=80,win_length=240)
    stft=np.reshape(stft,[-1,513])
    psd_all = np.abs(stft)**2

    
    psd = psd_all[300]

    fft_size = 2*(psd.shape[0]-1)

    order = 48


    #freq_axis = np.exp(np.linspace(np.log(1e-20),np.log(np.pi),fft_size//2+1))
    # 3 1-order glottal
    psd_res = psd
    for i in range(3):
        lpc,g = lpc_python(psd_res,order= 1)
        freq_axis,freqz_response=freqz(g,lpc,fft_size//2+1)
        re_psd =np.power(np.abs(freqz_response),2)
        psd_res = np.power(np.sqrt(psd_res)/np.sqrt(re_psd),2)
    

    # 48-order vocal tract 
    lpc,g = lpc_python(psd_res,order= order)
    freq_axis,freqz_response=freqz(g,lpc,fft_size//2+1)
    re_psd =np.power(np.abs(freqz_response),2)

    gross_vocal_psd = re_psd

    glottal_psd = psd/gross_vocal_psd

    # final 3-order glottal

    glottal_lpc,glottal_g = lpc_python(glottal_psd,order= 3)
    print(glottal_lpc.shape)
    freq_axis,freqz_response=freqz(glottal_g,glottal_lpc,fft_size//2+1)
    re_glottal_psd =np.power(np.abs(freqz_response),2)

    vocal_psd = psd/re_glottal_psd

    # final  vocal tract
    vt_lpc,vt_g = lpc_python(vocal_psd,order= order)
    freq_axis,freqz_response=freqz(vt_g,vt_lpc,fft_size//2+1)
    re_vt_psd =np.power(np.abs(freqz_response),2)


    plot_num=-1
    plt.subplot(5,1,1)
    plt.plot(10*np.log10(re_glottal_psd[:plot_num]))
    plt.subplot(5,1,2)
    plt.plot(10*np.log10(re_vt_psd[:plot_num]))
    plt.subplot(5,1,3)
    plt.plot(10*np.log10(re_glottal_psd[:plot_num]*re_vt_psd[:plot_num]))
    plt.subplot(5,1,4)
    plt.plot(10*np.log10(gross_vocal_psd[:plot_num]))
    plt.subplot(5,1,5)
    plt.plot(10*np.log10(psd[:plot_num]))
    plt.savefig("sp_res_stft.png")

    print(re_glottal_psd[:20])

if __name__ == '__main__':
    main()
