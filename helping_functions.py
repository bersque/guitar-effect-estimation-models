from scipy.io import wavfile
import numpy as np

sample_rate, wav = wavfile.read("C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/samples/looper_2.wav") ###zwraca wektor numpy-owy z probkami dzwieku  
sample_rate1, wav1 = wavfile.read("C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/samples/looper_2_d.wav") ###z efektrm, na gorze bez 

wav = wav.astype(np.float64)
wav1 = wav1.astype(np.float64)

def create_samples(string_name1,string_name2):
    sample_rate, wav = wavfile.read(f"C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/samples/{string_name1}.wav") ###zwraca wektor numpy-owy z probkami dzwieku  
    sample_rate1, wav1 = wavfile.read(f"C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/samples/{string_name2}.wav") 

    wav = wav.astype(np.float64)
    wav1 = wav1.astype(np.float64)


    wav,wav1 = match_signal(wav,wav1,22)

    return(wav,wav1,sample_rate,sample_rate1)
def export(signal,wav=wav,wav1=wav1):
    signal = np.clip(signal, -32768, 32767).astype(np.int16) ##tracimy odrobine informacji z konwersji na int ale tak malo ze pomijalne, inaczej wav sie nie kompiluje
    wav1 = np.clip(wav1, -32768, 32767).astype(np.int16)
    wav = np.clip(wav, -32768, 32767).astype(np.int16)

    wavfile.write("C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/results/estimated_effect.wav", sample_rate, signal)
    wavfile.write("C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/results/effect_original.wav", sample_rate, wav1)
    wavfile.write("C:/Users/Kacper/Desktop/programowanie/guitar_effect_estimator/results/original.wav", sample_rate, wav)

    print("File exported! :)")


def match_signal(s1,s2,dlugosc, detection_treshold = 400): #sygnal zaczyna sie od paru sekund ciszy, aby być w stanie zsynchronizować początki dla obu wektorów, dlugosc w sekundach
    dlugosc = dlugosc * sample_rate
    start_s1 = 0
    start_s2 = 0
    for j in range(0,len(s1)):
        if abs(s1[j]) > detection_treshold:
            start_s1 = j
            break
    for j in range(0,len(s2)):
        if abs(s2[j]) > detection_treshold:
            start_s2 = j
            break
    return(s1[start_s1:(start_s1+dlugosc)],s2[start_s2:(start_s2+dlugosc)]) ### zwraca 2 zsynchronizowane czasowo wektory 

def MSLE(s1,s2):
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    n = len(s1)
    squares = [(x-y)**2 for x,y in zip(s1,s2)]
    return sum(squares)/n