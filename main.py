from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from sympy import *
from helping_functions import *
from delay_lsm import lsm_delay, delay, cascading_delay
from stochastic_distortion import *
from reverb import *

print("Program started!...")

wav,wav1,sample_rate,sample_rate1 = create_samples("fsc","fsd")
wav2,wav3,sample_rate2,sample_rate3 = create_samples("kuroi_s1","u_t_b_s1")

def main_delay_lsm(s1,s2): ## s1 - probka, s2 - probka+delay
    coeff = lsm_delay(s1,s2)
    estimated_track = delay(s1,sample_rate,coeff[0],coeff[1])
    err = MSLE(s2,estimated_track)
    print(f"y^(x) = y(x) + {round(coeff[1],2)}*y(x-{round(coeff[0],2)})\n")
    print(f"Błąd średniokwadratowy: {err}.\n") 
    export(estimated_track)

def main_dist_atan(s1,s2):
    coeff = gradient_descent_stochastic_noa(gradient,10000,0.8,wav,wav1)
    estimated_track=distort_atan(wav,coeff[0],coeff[1],wav,wav1)
    err = MSLE(s2,estimated_track)
    print(f"y = {coeff[0]}*arctan({coeff[1]}x)")
    print(f"MSE: {err}")
    export(estimated_track,wav,wav1)

def main_reverb(s1,s2):
    h_est, wav_fft = reverb_fft(s1,s2) #estymator odpowiedzi impulsowej
    estimated_fft = h_est*wav_fft ## "splot" (jesteśmy nadal w frq domain)
    estimated_track = ifft(estimated_fft) ## powrot do orginalnej dziedziny

    err = MSLE(s2,estimated_track)
    print(f"MSE: {err}") ##absurdalnie niskie mse!
    export(estimated_track,wav,wav1)