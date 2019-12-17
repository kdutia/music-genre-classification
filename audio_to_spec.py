# Functions for converting audio to spectrograms.
# TODO: set config using 'conf' class

import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython

class audio_to_spec():
    def __init__(self, fmin, fmax, n_fft, sampling_rate=44100, hop_length=256, n_mels=128):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

        # min and max frequency for displaying spectrogram
        self.fmin = fmin or 20 # default 20Hz == lowest freq for human hearing
        self.fmax = fmax or sampling_rate // 2 # default nyquist-shannon

        # height of returned spectrogram matrix
        self.n_mels = n_mels

        # length of fft window
        self.n_fft = n_fft or n_mels*20 # TODO: is this a sensible default?

        # TODO:
        #self.duration = duration : useful if we want to clip audio files to a certain length
        #self.samples = samples : could also specify number of samples we want to have in the spectrogram


    def read_audio(self, pathname):
        y, sr = librosa.load(pathname, sr=self.sampling_rate)
        return y

    def audio_to_melspectrogram(self, audio):
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

    def show_melspectrogram(self, mels, title='Log-frequency power spectrogram'):
        librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                                sr=conf.sampling_rate, hop_length=conf.hop_length,
                                fmin=conf.fmin, fmax=conf.fmax
                                )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()    

    def read_as_melspectrogram(self, pathname, debug_display=False):
        x = self.read_audio(pathname)
        mels = self.audio_to_melspectrogram(x)
        if debug_display:
            IPython.display.display(IPython.display.Audio(x, rate=44100))
            self.show_melspectrogram(mels)
        return mels

    def audio_file_to_melspec_file(self, audio_path:str, image_path:str):
        mels = self.read_as_melspectrogram(audio_path)

        # TODO: normalize/convert to colour?

        matplotlib.image.imsave(image_path, mels)
