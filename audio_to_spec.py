# Functions for converting audio to spectrograms.
import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython

class audio_to_spec():
    def __init__(self, fmin=None, fmax=None, n_fft=None, sampling_rate=44100, hop_length=256, n_mels=128, duration=None):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

        # min and max frequency for displaying spectrogram
        self.fmin = fmin or 20 # default 20Hz == lowest freq for human hearing
        self.fmax = fmax or sampling_rate // 2 # default nyquist-shannon

        # height of returned spectrogram matrix
        self.n_mels = n_mels

        # length of fft window
        self.n_fft = n_fft or n_mels*20 # TODO: is this a sensible default?
        
        # force certain duration of audio files (seconds)
        self.duration = duration 

        # TODO:
        #self.samples = samples : could also specify number of samples we want to have in the spectrogram


    def read_audio(self, pathname):
        if self.duration is not None:
            # width of y = sr * duration
            y, sr = librosa.load(pathname, sr=self.sampling_rate, duration=self.duration)
        else:
            y, sr = librosa.load(pathname, sr=self.sampling_rate)
        return y

    def audio_to_melspectrogram(self, audio):
        spectrogram = librosa.feature.melspectrogram(audio, 
                                                    sr=self.sampling_rate,
                                                    n_mels=self.n_mels,
                                                    hop_length=self.hop_length,
                                                    n_fft=self.n_fft,
                                                    fmin=self.fmin,
                                                    fmax=self.fmax
                                                    )
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram

    def show_melspectrogram(self, mels, title='Log-frequency power spectrogram'):
        librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                                sr=self.sampling_rate, hop_length=self.hop_length,
                                fmin=self.fmin, fmax=self.fmax
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


def audio_to_spec_batch(path_to_csv):
    """convert audio files to spectrogram pngs in the same folder using a csv to find all the audio files"""

    df = pd.read_csv(path_to_csv, index_col=0)

    print('converting audio files to spectrogram images..')

    atspec = audio_to_spec(duration=30)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        audio_path = os.path.join(str(path),'data',row['relative_dir_wav'])
        img_path = audio_path[0:-4] + '.png'

        atspec.audio_file_to_melspec_file(audio_path, img_path)


if __name__ == "__main__":
    from tqdm import tqdm
    import pandas as pd
    from pathlib import Path
    import os
    
    path = Path.cwd()
    csv_path = path/'data'/'wav_files.csv'

    audio_to_spec_batch(csv_path)