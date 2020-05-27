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
    wave_data.shape = -1, params[0]
    wave_data = wave_data.T
    # print(len(wave_data))
    # params: channel number, sample width, sample rate
    return wave_data, r_params


def sound(s, params):
    channels = params[0]
    sampwidth = params[1]
    Fs = params[2]
    chunk = 1024
    s = s.T
    s.shape = 1, -1
    str_sound = s.tostring()
    nsamples = len(str_sound)
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sampwidth),
                    channels=channels, rate=Fs, output=True)
    for i in range(0, nsamples, chunk):
        stream.write(str_sound[i:i+chunk])
    leftNum = len(s) % chunk
    if leftNum != 0:
        stream.write(str_sound[nsamples-leftNum:])


def audiowrite(filename, wave_data, params):
    channels = params[0]
    sampwidth = params[1]
    Fs = params[2]
    f = wave.open(filename, 'wb')
    f.setnchannels(channels)
    f.setsampwidth(sampwidth)
    f.setframerate(Fs)
    wave_data = wave_data.T
    wave_data.shape = 1, -1
    f.writeframes(wave_data.tostring())
    f.close()


class Recorder:
    def __init__(self, channels, rate, frameTime, paddingTime):
        self.format = pyaudio.paInt16
        self.frameLen = int(frameTime/1000*rate)
        self.paddingLen = int(paddingTime/1000*rate)
        self.frameNum = int(1500/(frameTime-paddingTime))
        self.chunk = self.frameLen + (self.frameNum-1)*(self.frameLen-self.paddingLen)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format, channels=channels, rate=rate, input=True, frames_per_buffer=self.chunk)
        print("Initializing...")
        for i in range(3):
            stringData = self.stream.read(self.chunk)
            self.data = np.fromstring(stringData, dtype=np.short)
        print("Recorder Initialize finished!")
        # self.data = stringData
        self.currentFrame = 0

    def getFrame(self):
        if self.currentFrame + self.frameLen <= self.chunk:
            frame = self.data[self.currentFrame:self.currentFrame+self.frameLen]
            self.currentFrame = self.currentFrame + self.frameLen
            return frame
        else:
            stringData = self.stream.read(self.chunk)
            self.data = np.fromstring(stringData, dtype=np.short)
            # print("Recording...")
            self.currentFrame = 0
            return self.data[self.currentFrame:self.currentFrame+self.frameLen]

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


if __name__ == '__main__':
    s, p = audioread(r'.\audios\left.wav')
    print(p)
    sound(s, (1, 2, 8000))