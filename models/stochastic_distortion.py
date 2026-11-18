from helpers.helping_functions import * 
import numpy as np
import random
from scipy.optimize import curve_fit

wav,wav1,sample_rate,sample_rate1 = create_samples("fsc","fsd")
###probujemy zaimplementowac SGD dla funkcji 2 zmiennych (n zmiennych?)
###probujemy najpierw stochastyczny, ponieważ mamy absurdalną ilość danych

def gradient(vec,bez_efektu,z_efektem): #gradient dla funkcji celu dla a*arctan(x*k)
    a = vec[0]
    k = vec[1]
    x = bez_efektu
    y = z_efektem
    da = (np.arctan(x*k))*(a*np.arctan(x*k) - y)
    dk = ((x*a)/(1+(x*k)**2))*(a*np.arctan(x*k) - y)
    return np.array([da,dk])

def gradient_descent_stochastic(gradient_funkcja,iteracje,eta,proba_bez_efektu,proba_z_efektem,wektor_startowy = [1,2]): ##dziala okropnie, bo losujemy uniformly
    wektor = np.array(wektor_startowy)
    indeksy = np.arange(len(proba_bez_efektu)) ## wektor z ktorego losujemy indeksy

    proba_bez_efektu = proba_bez_efektu / 32768
    proba_z_efektem = proba_z_efektem / 32768

    for i in range(0,iteracje):
        losowy_indeks = random.choice(indeksy)
        xobs = proba_z_efektem[losowy_indeks]
        xorg = proba_bez_efektu[losowy_indeks]
        wektor = wektor - eta*gradient_funkcja(wektor,xorg,xobs)
        reszta_modelu = (xobs - wektor[0]*np.arctan(xorg*wektor[1]))**2 
        wartosc_funkcji = wektor[0]*np.arctan(xorg*wektor[1])
        print(wektor, reszta_modelu, wartosc_funkcji)
    wektor[0] = 32768 * wektor[0]
    wektor[1] = wektor[1] / 32768
    return wektor

def distort_atan(signal,a,k,wav,wav1): ###naklada distortion
    peak = max(np.abs(signal).max(), np.abs(wav).max(), np.abs(wav1).max())
    if peak > 32767: #nie chcemy stworzyć przesteru ani innych artefaktow, w razie czego skalujemy lekko w dół
        gain = 32767 / peak
    else:
        gain = 1.0

    signal_out = ((a*np.arctan(k*signal)) * gain).astype(np.int16)
    return(signal_out)


def main(s1,s2): ## s1 - probka, s2 - probka+dist
    coeff = gradient_descent_stochastic(gradient,iteracje=20000,eta=0.01,proba_bez_efektu=s1, proba_z_efektem=s2)
    estimated_track = distort_atan(s1,coeff[0],coeff[1],wav,wav1)
    err = RMSE(s2,estimated_track)
    print(f"y^(x) = {coeff[0]}*arctan({coeff[1]}*x)")
    print(f"Błąd średniokwadratowy: {err}.\n") 
    export(estimated_track)


def model(x, a, k):
    return a * np.arctan(k * x)
def sci_dist():
    popt, _ = curve_fit(model, wav/32768, wav1/32768, p0=[1, 2], maxfev=20000)
    a, k = popt
    a *= 32768
    k /= 32768
    estimated_track = distort_atan(wav,a,k,wav,wav1)

    err = RMSE(wav1,estimated_track)
    print(f"y = {a}*arctan({k}x)")
    print(f"MSE: {err}")

    export(estimated_track,wav,wav1)

def gradient_descent_stochastic_weighted(gradient_funkcja,iteracje,eta,proba_bez_efektu,proba_z_efektem,wektor_startowy = [1,2]): ##dziala okropnie, bo losujemy uniformly
    wektor = np.array(wektor_startowy)
    indeksy = np.arange(len(proba_bez_efektu)) ## wektor z ktorego losujemy indeksy

    proba_bez_efektu = proba_bez_efektu / 32768
    proba_z_efektem = proba_z_efektem / 32768

    wagi = np.abs(proba_bez_efektu)
    prawdopodobienstwa = wagi / wagi.sum()

    for i in range(0,iteracje):
        losowy_indeks = np.random.choice(indeksy, size=1, replace=False, p=prawdopodobienstwa)[0] ##zmien size aby zrobic batche!
        xobs = proba_z_efektem[losowy_indeks]
        xorg = proba_bez_efektu[losowy_indeks]
        wektor = wektor - eta*gradient_funkcja(wektor,xorg,xobs)
        reszta_modelu = (xobs - wektor[0]*np.arctan(xorg*wektor[1]))
        wartosc_funkcji = wektor[0]*np.arctan(xorg*wektor[1])
        print(wektor, reszta_modelu, wartosc_funkcji, i)
    wektor[0] = 32768 * wektor[0]
    wektor[1] = wektor[1] / 32768
    return wektor


def gradient_descent_stochastic_noa(gradient_funkcja, iteracje, eta, proba_bez_efektu, proba_z_efektem,wektor_startowy=[1, 2]):
    wektor_startowy = list(wektor_startowy)  # probujemy wylapac a z samej amplitudy
    proba_z_efektem_norm_tmp = proba_z_efektem / 32768
    wektor_startowy[0] = np.max(np.abs(proba_z_efektem_norm_tmp)) / (np.pi / 2)  

    wektor = np.array(wektor_startowy, dtype=np.float64)
    indeksy = np.arange(len(proba_bez_efektu))

    proba_bez_efektu = proba_bez_efektu / 32768
    proba_z_efektem = proba_z_efektem / 32768

    wagi = np.abs(proba_bez_efektu)
    prawdopodobienstwa = wagi / wagi.sum()

    

    losowe_indeksy = np.random.choice(indeksy,size=iteracje,replace=True,p=prawdopodobienstwa)

    for i, j in enumerate(losowe_indeksy):
        xobs = proba_z_efektem[j]
        xorg = proba_bez_efektu[j]

        grad = gradient_funkcja(wektor, xorg, xobs)
        wektor[1] -= eta * grad[1]
        if i % 1000 == 0:
            print(i, wektor)

    wektor[0] = 32768 * wektor[0]
    wektor[1] = wektor[1] / 32768
    return wektor
