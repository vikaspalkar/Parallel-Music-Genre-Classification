from music.feature import *
from STFT import stft
import numpy as np


def spectral_flux(wavedata, window_size, sample_rate):
    import numpy as np
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    timebins, freqbins = np.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))

    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum))**2, axis=1)) / freqbins

    return sf[1:] #, np.asarray(timestamps)