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
    x=np.array(x,np.float64)
    _f0,t=pyworld.dio(x,fs)
    f0=pyworld.stonemask(x,_f0,t,fs)
    sp=pyworld.cheaptrick(x,f0,t,fs)

    psd = sp[200]

    fft_size = 2*(psd.shape[0]-1)

    order = 48
    lpc,g = lpc_python(psd,order= order)

    roots = np.roots(lpc)


    roots_select = roots[np.imag(roots) >= 0]

    angz = np.arctan2(np.imag(roots_select),np.real(roots_select))

    freqs = angz*fs/(2*np.pi)

    #freqs = sorted(angz*fs/(2*np.pi))
    bws = [-0.5*fs/(2*np.pi)*np.log(np.abs(roots_select[i])) for i,f in enumerate(freqs) ]

    for freq,bw in zip(freqs,bws):
        if freq>90 and bw < 400:
            print(freq/8000*512,freq,bw)
    print(freqs)
    print(bws)
    print(np.array(freqs)/8000*512)




    freq_axis,freqz_response=freqz(g,lpc,fft_size//2+1)
    freq_axis=freq_axis/np.pi*fs/2
    resp=np.power(np.abs(freqz_response),2)
    resp2 = np.power(g/np.abs(np.fft.rfft(lpc,fft_size)),2)

    print(np.allclose(resp,resp2))
    
    ratio=np.sum(resp)/np.sum(psd)
    print(ratio)

    plt.subplot(3,1,1)
    plt.plot(10*np.log10(resp))
    plt.subplot(3,1,2)
    plt.plot(10*np.log10(resp2))
    plt.subplot(3,1,3)
    #plt.xticks(np.arange(0,fs/2,500))
    plt.plot(freq_axis,10*np.log10(psd))
    plt.savefig("resp.png")


if __name__ == '__main__':
    main()
