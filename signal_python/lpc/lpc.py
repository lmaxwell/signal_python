import numpy as np
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

def psd2lpc(psd, order):
    # N: compute next power of 2
    r = np.fft.irfft(psd)[:order+1]
    lpc,_,_ = levinson(r, order)
    g = np.sqrt(np.sum(lpc * r))
    return lpc,g
def lpc2psd(lpc,g,fft_size):
    psd = np.power(g/np.abs(np.fft.rfft(lpc,fft_size)),2)
    return psd