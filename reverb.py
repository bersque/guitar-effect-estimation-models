from helping_functions import * 
import numpy as np
from scipy.optimize import curve_fit
from numpy.fft import fft, ifft


print("Program started...")

wav,wav1,sample_rate,sample_rate1 = create_samples("looper_2","looper_2_r")

def reverb_fft(s1,s2):
    wav_fft = np.array(fft(wav))
    wav1_fft = np.array(fft(wav1))
    
    est = wav1_fft/wav_fft ## estymator odpowiedzi impulsowej w frq domain

    return est, wav_fft