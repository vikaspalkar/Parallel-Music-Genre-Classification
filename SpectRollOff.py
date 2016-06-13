from music.feature import *

from STFT import stft
import numpy as np

def spectral_rolloff(wavedata, window_size, sample_rate, k=0.85):
    import numpy as np
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    power_spectrum     = np.abs(magnitude_spectrum)**2
    timebins, freqbins = np.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))

    sr = []

    spectralSum    = np.sum(power_spectrum, axis=1)

    for t in range(timebins-1):

        # find frequency-bin indeces where the cummulative sum of all bins is higher
        # than k-percent of the sum of all bins. Lowest index = Rolloff
        sr_t = np.where(np.cumsum(power_spectrum[t,:]) >= k * spectralSum[t])[0][0]

        sr.append(sr_t)

    sr = np.asarray(sr).astype(float)

    # convert frequency-bin index to frequency in Hz
    sr = (sr / freqbins) * (sample_rate / 2.0)

    return sr #, np.asarray(timestamps)

