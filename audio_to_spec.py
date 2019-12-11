# Functions for converting audio to spectrograms.
# TODO: set config using 'conf' class

import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython

class conf:
    """Config class"""
    sampling_rate = 44100
    duration = 2
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration


def read_audio(pathname):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    return y

def audio_to_melspectrogram(audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax
                                                )
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_melspectrogram(mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                             fmin=conf.fmin, fmax=conf.fmax
                            )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()    

def read_as_melspectrogram(pathname, debug_display=False):
    x = read_audio(pathname)
    mels = audio_to_melspectrogram(x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=44100))
        show_melspectrogram(mels)
    return mels

def audio_file_to_melspec_file(audio_path:str, image_path:str):
    mels = read_as_melspectrogram(audio_path)

    # TODO: normalize/convert to colour?

    matplotlib.image.imsave(image_path, mels)
