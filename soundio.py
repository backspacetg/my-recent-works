import numpy as np
import wave
import pyaudio


def audioread(filename):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    r_params = params[:3]
    nsamples = params[3]
    str_data = f.readframes(nsamples)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    # print(len(wave_data))
    return wave_data, r_params


def sound(s, params):
    channels = params[0]
    sampwidth = params[1]
    Fs = params[2]
    nsamples = len(s)
    chunk = 1024
    str_sound = s.tostring()
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sampwidth),
                    channels=channels, rate=Fs, output=True)
    for i in range(0, nsamples, chunk):
        stream.write(str_sound[i:i+chunk])
    leftNum = len(s) % chunk
    if leftNum != 0:
        stream.write(str_sound[nsamples-leftNum:])




if __name__ == '__main__':
    s, p = audioread('test.wav')
    # c, w, fs, n = p
    sound(s, p[0:3])