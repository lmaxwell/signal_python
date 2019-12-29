import numpy as np
from scipy.signal import freqz
from signal_python.lpc import lpc 

def gmf_iaif(psd,glottal_order=3,vt_order=48,):
    """
    psd : 1-d numpy array, shape : fft_size/2 +1
    """
    fft_size=(psd.shape[0]-1)*2
    psd_res = psd
    for i in range(3):
        lpc_coef,g = lpc.psd2lpc(psd_res,order= 1)
        re_psd = lpc.lpc2psd(lpc_coef,g,fft_size)
        psd_res = np.power(np.sqrt(psd_res)/np.sqrt(re_psd),2)
    

    # vt_order vocal tract 
    lpc_coef,g = lpc.psd2lpc(psd_res,order= vt_order)
    re_psd = lpc.lpc2psd(lpc_coef,g,fft_size)

    gross_vocal_psd = re_psd

    glottal_psd = psd/gross_vocal_psd

    # final glottal_order glottal

    glottal_lpc,glottal_g = lpc.psd2lpc(glottal_psd,order= glottal_order)
    re_glottal_psd = lpc.lpc2psd(glottal_lpc,glottal_g,fft_size)

    vocal_psd = psd/re_glottal_psd

    # final  vocal tract
    vt_lpc,vt_g = lpc.psd2lpc(vocal_psd,order= vt_order)
    #re_vt_psd = lpc.lpc2psd(vt_lpc,vt_g,fft_size)

    return glottal_lpc,glottal_g,vt_lpc,vt_g