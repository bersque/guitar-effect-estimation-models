from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import math
import os
from scipy.signal import correlate, correlation_lags
from scipy.signal import correlate, correlation_lags, find_peaks
from sympy import *
from helping_functions import *
from delay_lsm import lsm_delay, delay, cascading_delay
from stochastic_distortion import *

print("Program started!...")

wav,wav1,sample_rate,sample_rate1 = create_samples("fsc","fsd")

def main_delay_lsm(s1,s2): ## s1 - probka, s2 - probka+delay
    coeff = lsm_delay(s1,s2)
    estimated_track = delay(s1,sample_rate,coeff[0],coeff[1])
    err = MSLE(s2,estimated_track)
    print(f"y^(x) = y(x) + {round(coeff[1],2)}*y(x-{round(coeff[0],2)})\n")
    print(f"Błąd średniokwadratowy: {err}.\n") 
    export(estimated_track)

def main_dist_atan(s1,s2):
    coeff = gradient_descent_stochastic_noa(gradient,5000,0.8,wav,wav1)
    estimated_track=distort_atan(wav,coeff[0],coeff[1],wav,wav1)
    err = MSLE(s2,estimated_track)
    print(f"y = {coeff[0]}*arctan({coeff[1]}x)")
    print(f"MSE: {err}")
    export(estimated_track,wav,wav1)

main_dist_atan(wav,wav1) ## ALLELUJA, nadal trzeba popracowac ale jest super






