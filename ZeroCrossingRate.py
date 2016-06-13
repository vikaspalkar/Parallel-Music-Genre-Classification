#from music.feature import *
from STFT import stft
import numpy as np


def zero_crossing_rate(wavedata, block_length, samplerate):
    import numpy as np
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata)/block_length))

    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(samplerate)))

    zcr = []

    for i in range(0,num_blocks-1):

        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])

        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))
        zcr.append(zc)

    return np.asarray(zcr) #, np.asarray(timestamps)