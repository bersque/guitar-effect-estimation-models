import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from scipy.signal import correlate, correlation_lags, find_peaks
from sympy import *
from helping_functions import *

wav,wav1,sample_rate,sample_rate1 = create_samples("looper_2","looper_2_d")


def delay(signal, sample_rate, delay_seconds, alpha): #delay to orginalny sygnal + orginalny sygnal opozniony i zmniejszony, model z jednym pogłosem 
    peak = max(np.abs(signal).max(), np.abs(wav).max(), np.abs(wav1).max())
    if peak > 32767: #nie chcemy stworzyć przesteru ani innych artefaktow, w razie czego skalujemy lekko w dół
        gain = 32767 / peak
    else:
        gain = 1.0

    delay_rate = round(sample_rate * delay_seconds) 
    echo = np.zeros_like(signal)

    for i in range(delay_rate,len(signal)): ### najpierw tworzymy wektor zerowy, potem zapełniamy go przesunięciem
        echo[i] = alpha*signal[i-delay_rate]

    signal_out = ((signal+echo) * gain).astype(np.int16)
    return(signal_out)


####################################### korelacja wzajemna ################################################
###Poniższe wykrywa o ile opozniony jest poglos, przesuwamy fale wzgledem siebie o n jednostek i sprawdzamy korelacje wzajemną, najwieksza wygrywa 
def normalized_cross_correlation(x, y): 
    y = y-x ### zostaje sam delay w probce; inaczej program glupieje dla alfa < 1.
    x = (x - np.mean(x)) / np.std(x)  #normalizacja, usuwa zbytni wpływ amplitud na korelacje 
    y = (y - np.mean(y)) / np.std(y)
    return correlate(x, y, mode='full',method="fft") / len(x) ##wazne mieć włączone fft, inaczej tragicznie długo wylicza (O(n^2))

def cross_cor(x, y):
    ncc = normalized_cross_correlation(x, y)
    lags = correlation_lags(len(x), len(y), mode="full")

    best_idx = np.argmax((ncc))
    best_lag = lags[best_idx]

    delay_sec = best_lag / sample_rate
    return(abs(best_lag))

def lsm_delay(s1,s2,k=10000): ### s1 - sygnal orginalny, s2 - sygnal po delayu, zwracamy wspolczynniki do konstrukcji estymowanego delayu

    s1 = s1/k ##glupie bo glupie ale zapobiega przepełnieniu liczbowemu przy sumacji (k skraca się we wzorze wiec nie trzeba przemnazac spowrotem)
    s2 = s2/k

    d = cross_cor(s1,s2)
    delayed_s1 = np.zeros_like(s1)
    for i in range(d,len(s1)):
        delayed_s1[i] = s1[i-d]
    licznik = [x*(y-z) for x,y,z in zip(delayed_s1,s2,s1)]
    mianownik = [x**2 for x in delayed_s1]
    licznik_suma = sum(licznik)
    mianownik_suma = sum(mianownik)
    alfa = licznik_suma/mianownik_suma

    print(f"delay: {d/sample_rate}s, alfa: {alfa}\n")
    return [d/sample_rate,alfa] #krotka - delay, alfa 

####delay wielokrotny:

def cascading_delay(s1,s2,treshhold,create=True):
    ####this version asumes equal delay periods between echoes but works!!
    ##niedoszacowuje alfe dla kolejnych powtorzen, fragment z odejmowaniem od s2 srednio dziala
    ###to be fixed!!!!
    alpha_list = []
    check = True
    orginal_track = s1
    track_to_return = orginal_track
    while check:
        coeff = lsm_delay(s1,s2)
        if coeff[1] < treshhold or len(alpha_list) > 3: 
            check = False
        else:
            alpha_list.append(coeff)
            estimated_track = delay(s1,sample_rate,coeff[0],coeff[1])
            s2 = s2 - estimated_track + orginal_track
    if create == True:
        for i in  range(0,len(alpha_list)):
            echo = delay(orginal_track,sample_rate,alpha_list[0][0]*(i+1),alpha_list[i][1]) - orginal_track
            track_to_return = track_to_return + echo
            print("echo added")
    return track_to_return