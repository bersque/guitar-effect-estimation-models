from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import math
import os
from scipy.signal import correlate, correlation_lags
from scipy.signal import correlate, correlation_lags, find_peaks
from sympy import *
from helping_functions import MSLE,export,match_signal
from delay_lsm import lsm_delay, delay, cascading_delay

print("Program started!\n")


sample_rate, wav = wavfile.read("C:/Users/Kacper/Desktop/samples/looper_2.wav") ###zwraca wektor numpy-owy z probkami dzwieku  
sample_rate1, wav1 = wavfile.read("C:/Users/Kacper/Desktop/samples/looper_2_d.wav") 

wav = wav.astype(np.float64)
wav1 = wav1.astype(np.float64)


wav,wav1 = match_signal(wav,wav1,22)

def main(s1,s2): ## s1 - probka, s2 - probka+delay
    coeff = lsm_delay(s1,s2)
    estimated_track = delay(s1,sample_rate,coeff[0],coeff[1])
    err = MSLE(s2,estimated_track)
    print(f"y^(x) = y(x) + {round(coeff[1],2)}*y(x-{round(coeff[0],2)})\n")
    print(f"Błąd średniokwadratowy: {err}.\n") 
    export(estimated_track)

main(wav,wav1)