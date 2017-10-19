import os
import numpy as np
from preprocess_data import *
from keras.models import load_model
import pyaudio
import wave
import requests
import time
import json


def main():
    files = os.listdir('Preproc')
    mel_dims = [1, 1, 96, 366]
    classes = ['qaaf', 'sa', 'seen', 'zal', 'jeem', 'sheen', 'ba', 'zuaa', 'meem', 'suad', 'noon', 'tuaa', 'ta', 'yaa',
               'raa', 'zaa', 'kaaf', 'kha', 'ayn', 'ha', 'laam', 'waaw', 'haaa', 'zuad', 'ghayn', 'dal', 'alif', 'faa']

    X_test = np.zeros((len(files), mel_dims[1], mel_dims[2], mel_dims[3]))

    test_count = 0
    for infile in files:
        audio_path = os.path.join('Preproc', infile)
        melgram = np.load(audio_path)
        zeros = np.zeros(mel_dims)
        zeros[:, :, 0:melgram.shape[2], 0:melgram.shape[3]] = melgram
        melgram = zeros
        X_test[test_count, :, :] = melgram
        test_count += 1

    model = load_model('my_model.h5')
    pred = model.predict(X_test, verbose=0)

    for f in files:
        idx = np.argmax(pred[files.index(f)])
        label = classes[idx]
        print(f, ' --> ', label)

def record_audio(RECORD_SECONDS, WAVE_OUTPUT_FILENAME):
    # --------- SETTING PARAMS FOR OUR AUDIO FILE ------------#
    FORMAT = pyaudio.paInt16  # format of wave
    CHANNELS = 2  # no. of audio channels
    RATE = 44100  # frame rate
    CHUNK = 1024  # frames per audio sample
    # --------------------------------------------------------#

    # creating PyAudio object
    audio = pyaudio.PyAudio()

    # open a new stream for microphone
    # It creates a PortAudio Stream Wrapper class object
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    # ----------------- start of recording -------------------#
    print("Listening...")

    # list to save all audio frames
    frames = []

    for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
        # read audio stream from microphone
        data = stream.read(CHUNK)
        # append audio data to frames list
        frames.append(data)

    # ------------------ end of recording --------------------#
    print("Finished recording.")

    stream.stop_stream()  # stop the stream object
    stream.close()  # close the stream object
    audio.terminate()  # terminate PortAudio

    # ------------------ saving audio ------------------------#

    # create wave file object
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')

    # settings for wave file object
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))

    # closing the wave file object
    waveFile.close()

if __name__ == '__main__':
    # record_audio(3,"Samples/test.wav")
    preprocess_dataset()
    exit(main())