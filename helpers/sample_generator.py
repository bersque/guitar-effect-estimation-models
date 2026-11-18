from models.delay_lsm import *
from models.reverb import *
from models.stochastic_distortion import *
from helpers.helping_functions import *
import os
from pathlib import Path
from scipy.io import wavfile
import numpy as np

def harmonic_generator(n, dlugosc):
    rng = np.random.default_rng()

    sr = 44100
    t = np.arange(sr * dlugosc) / sr

    f0 = rng.uniform(70, 500)

    x = np.zeros(len(t))

    for i in range(1, n + 1):

        amplitude = rng.uniform(1000, 3000) / i
        phase = rng.uniform(0, 2*np.pi)

        x += amplitude * np.sin(
            2 * np.pi * f0 * i * t + phase
        )

    return x        

wav_r = harmonic_generator(20,10)

def short_export(signal,name):
    signal = np.clip(signal, -32768, 32767).astype(np.int16)
    wavfile.write(f"C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/benchmark/dist/{name}.wav", sample_rate, signal)
    print(f"{name} exported!")

def benchmark_delete():
    folder = Path("C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/benchmark/dist").resolve()
    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)
        print(f"Deleted: {folder}")
    else:
        print("Folder does not exist.")

def short_export_effect(signal,name,parameters,mode):
    if mode == 'single_delay':
        signal = delay(signal,sample_rate,parameters[0],parameters[1])
    elif mode == 'distortion':
        signal = distort_atan(signal,parameters[0],parameters[1],wav,wav1)
    elif mode == 'reverb':
        signal = ifft(signal*parameters[0])
    signal = np.clip(signal, -32768, 32767).astype(np.int16)
    wavfile.write(f"C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/benchmark/dist/{name}_e.wav", sample_rate, signal)
    print(f"{name} exported!")

def mass_production(n, mode):
    newpath = r'C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/benchmark/dist' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    rng = np.random.default_rng()
    del_s = rng.uniform(0,1.5)
    alpha = rng.uniform(0,1)
    a = None
    k = None
    h = None
    for i in range(0,n):
        no_harmonics = rng.integers(5,30)
        length = rng.uniform(2,10)
        sound = harmonic_generator(no_harmonics,length)
        short_export(sound,i)
        if mode == 'single_delay':
            param = [del_s,alpha]
            print(del_s,alpha)
            short_export_effect(sound,i,param,mode)
        elif mode == 'distortion':
            param = [a,k]
            print(a,k)
            short_export_effect(sound,i,param,mode)
        elif mode == 'reverb':
            param = h
            print(h)
            short_export_effect(sound,i,param,mode)
