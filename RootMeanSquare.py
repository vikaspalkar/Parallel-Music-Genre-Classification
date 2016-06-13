#from music.feature import *
from STFT import stft
import numpy as np


def root_mean_square(wavedata, block_length, sample_rate):
    import numpy as np
    # how many blocks have to be processed?
    length =len(wavedata)

    num_blocks = int(np.ceil(length/block_length))

    #num_blocks= int(num_blocks)
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(sample_rate)))


    rms = []
    for i in range(0,num_blocks-1):

        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])

        rms_seg = np.sqrt(np.mean(wavedata[start:stop]**2))

        rms.append(rms_seg)

    return np.asarray(rms) #, np.asarray(timestamps)



