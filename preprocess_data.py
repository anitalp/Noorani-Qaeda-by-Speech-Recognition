from __future__ import print_function
import numpy as np
import librosa
import librosa.display
import shutil
import os

def preprocess_dataset(inpath="Samples/", outpath="Preproc/"):

    if not os.path.exists(outpath):
        os.mkdir(outpath);   # make a new directory for preproc'd files
    else:
        shutil.rmtree(outpath) # delete previous files in preproc'd
        os.mkdir(outpath)

    files = os.listdir(inpath)
    for infilename in files:
        audio_path = os.path.join(inpath, infilename)
        aud, sr = librosa.load(audio_path, sr=None)
        melgram = librosa.logamplitude(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]
        outfile = os.path.join(outpath, infilename + '.npy')
        np.save(outfile, melgram)