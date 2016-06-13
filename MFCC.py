from music.feature import *
import numpy as np


from scipy.io import loadmat
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct

#from scikits.talkbox import segment_axis

def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    import numpy as numpy
    from scipy.io import loadmat
    from scipy.signal import lfilter, hamming
    from scipy.fftpack import fft
    from scipy.fftpack.realtransforms import dct
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = numpy.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** numpy.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1,
                        numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        lslope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1,
                        numpy.floor(hi * nfft / fs) + 1, dtype=numpy.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs

def mfcc(input, nwin=256, nfft=512, fs=16000, nceps=13):
    import numpy as numpy
    from scipy.io import loadmat
    from scipy.signal import lfilter, hamming
    from scipy.fftpack import fft
    from scipy.fftpack.realtransforms import dct
    over = nwin - 160
    prefac = 0.97

    #lowfreq = 400 / 3.
    lowfreq = 133.33

    linsc = 200/3.
    logsc = 1.0711703

    nlinfil = 13
    nlogfil = 27
    nfil = nlinfil + nlogfil

    w = hamming(nwin, sym=0)

    fbank = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)[0]
    if fbank<=0:
        fbank=0.0001
    #------------------
    # Compute the MFCC
    #------------------
    extract = preemp(input, prefac)
    framed = segment_axis(extract, nwin, over) * w

    # Compute the spectrum magnitude
    spec = numpy.abs(fft(framed, nfft, axis=-1))

    if spec<=0:
        spec=0.00001

    # Filter the spectrum through the triangle filterbank
    arr= numpy.dot(spec,fbank.T)                                         ##CHANGED CODE
    print "LOG ARRAy =",arr
    mspec=numpy.log10(arr)
    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]

    return ceps

def preemp(input, p):
    from scipy.signal import lfilter, hamming
    """Pre-emphasis filter."""
    return lfilter([1., -p], 1, input)

def hz2mel(f):
    """Convert an array of frequency in Hz into mel."""
    import numpy as numpy
    return 1127.01048 * numpy.log(f/700 +1)

def mel2hz(m):
    import numpy as numpy
    """Convert an array of frequency in Hz into mel."""
    return (numpy.exp(m / 1127.01048) - 1) * 700

"""sgementaxis code.

This code has been implemented by Anne Archibald, and has been discussed on the
ML."""

import warnings

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):

    import numpy as numpy
    from scipy.io import loadmat
    from scipy.signal import lfilter, hamming
    from scipy.fftpack import fft
    from scipy.fftpack.realtransforms import dct
    if axis is None:
        a = numpy.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap < 0 or length <= 0:
        raise ValueError, "overlap must be nonnegative and length must "\
                          "be positive"

    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = numpy.empty(s,dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
        raise ValueError, \
              "Not enough data points to segment array in 'cut' mode; "\
              "try 'pad' or 'wrap'"
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return numpy.ndarray.__new__(numpy.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
                     + a.strides[axis+1:]
        return numpy.ndarray.__new__(numpy.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


def stEnergy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))