from models.delay_lsm import *
from models.reverb import *
from models.stochastic_distortion import *
from helpers.helping_functions import *
from pathlib import Path
from scipy.io import wavfile
import numpy as np
from helpers.sample_generator import *


def benchmark_MLE(lista_probek_bez_efektu,lista_probek_z_efektem,estymator):
    mle_list = []
    for i in range(0,len(lista_probek_bez_efektu) - 1):
        estimated_track = estymator(lista_probek_bez_efektu[i],lista_probek_z_efektem[i])   
        mle_list.append(RMSE(estimated_track,lista_probek_z_efektem[i]))
    return mle_list



def bench():
    benchmarks = []
    folder = Path("benchmark/dist")

    samples = {}

    for file in folder.glob("*.wav"):
        sample_rate, signal = wavfile.read(file)
        samples[file.stem] = signal
    for i in range(0,100):
        no_effect = samples[f'{i}'] 
        effect = samples[f'{i}_e']
        params = lsm_delay(no_effect,effect)
        for j in range(0,100):
            benchmarks.append((i,j,RMSE(samples[f'{j}_e'],delay(samples[f'{j}'],sample_rate,params[0],params[1]))))
        return benchmarks
    return benchmarks
        
        



benchmarkii = bench()
print(max(benchmarkii),min(benchmarkii))