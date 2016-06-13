                    #Python multi-processing... tried to run feature extraction on multi-cores


#from SpectCentroid import spectral_centroid
#from SpectRollOff import spectral_rolloff
#from STFT import stft
import csv
import warnings
from Tkinter import *
import time
import StringIO
from scipy.signal import lfilter, hamming
#from RootMeanSquare import root_mean_square
#from SpectFlux import spectral_flux
#from music.feature.MFCC import stEnergy
#from music.feature.ZeroCrossingRate import zero_crossing_rate
warnings.filterwarnings('ignore')
import pp
# numerical processing and scientific libraries
import numpy
from numpy import inf
# signal processing
from scipy.io                     import wavfile
#from scikits.talkbox.features     import mfcc
import glob
import threading
#from joblib import Parallel, delayed
from multiprocessing import Process, Queue,Pool

def ClassToInt(name):
        if name=="blues":
            return 0
        elif name=="classical":
            return 1
        elif name=="country":
            return 2
        elif name=="hiphop":
            return 3
        elif name=="jazz":
            return 4
        elif name=="metal":
            return 5
        elif name=="pop":
            return 6
def parFeatureExtraction(dataset_location,pathLength):
    t0=time.time()
    path =dataset_location+'/*.wav'
    files=glob.glob(path)
    auList=list()
    for file in files:
        auList.append(file)

    jobs=[]
    with open('/home/ubantu/TwoClassfeatureSet.csv', 'w') as csvfile:
        fieldnames = ['Spect Centroid', 'Spect Rolloff','Spect Flux','RMS','ZCR','SC_SD','SR_SD','SF_SD','ZCR_SD','energy',\
                      'MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11','MFCC12','MFCC13',\
                      'CLASS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        job_server = pp.Server()
        fileIndex=[]
        jobs = [(file, job_server.submit(extractFeature,(file,pathLength),\
                (ClassToInt,spectral_centroid,spectral_rolloff,spectral_flux,\
                 root_mean_square,zero_crossing_rate,mfcc,stEnergy,stft,trfbank,hz2mel,mel2hz,preemp,segment_axis),\
                ("numpy","wavfile","loadmat","lfilter","hamming","fft","dct",))) for file in auList]
        for input,job in jobs:
            #print "INPUT=  ",input,"  JOB=   ",job()
            fileIndex.append(input)
            result=job()
            s1=result[0]
            sr1=result[1]
            sf1=result[2]
            rms=result[3]
            zcr=result[4]
            MFCC_res=result[5]
            #MFCCs=MFCC_res[0]
            #rms= rms[~numpy.isnan(rms)] #rms array contains NAN values and we have to remove these values

            MFCC_coef=list()     #TEMP COMMENT
            ran=MFCC_res.shape
            #print "RAANNNN===",ran
            #for ind in range(len(MFCCs)):
            #    MFCCs[ind][MFCCs[ind] == -inf] = 0
            ran1=ran[0]
            for ind1 in range(13):    #TEMP COMMENT
                sum=0
                for ind in range(ran1):
                    sum+=MFCC_res[ind,ind1]
                MFCC_coef.append(sum/ran1)     #TEMP COMMENT
            eng=result[6]
            intClass=result[7]          #TEMP COMMENT
            #print result,"<===JOB"
            writer.writerow({'Spect Centroid':s1.mean().astype(float), 'Spect Rolloff':sr1.mean().astype(float),'Spect Flux':sf1.mean().astype(float),'RMS':rms.mean().astype(float),'ZCR':zcr.mean().astype(float),\
                            'SC_SD':s1.std().astype(float),'SR_SD':sr1.std().astype(float),'SF_SD':sf1.std().astype(float),'ZCR_SD':zcr.std().astype(float),'energy':eng.astype(float),\
                            'MFCC1':MFCC_coef[0],'MFCC2':MFCC_coef[1],'MFCC3':MFCC_coef[2],'MFCC4':MFCC_coef[3],\
                            'MFCC5':MFCC_coef[4],'MFCC6':MFCC_coef[5],'MFCC7':MFCC_coef[6],'MFCC8':MFCC_coef[7],\
                            'MFCC9':MFCC_coef[8],'MFCC10':MFCC_coef[9],'MFCC11':MFCC_coef[10],'MFCC12':MFCC_coef[11],\
                            'MFCC13':MFCC_coef[12],'CLASS':intClass})


    print "feature extraction done in=   ",time.time()-t0
    #joinCSVs(auList,pathLength)
    print job_server.print_stats()
    print "=======================================END"


def extractFeature(file,pathLength):
    import numpy as numpy
    from scipy.io import wavfile

    win=512
    string1= file
    cut=string1[pathLength:]
    #cut=string1[75:]
    cut=cut[:-10]
    intClass=ClassToInt(cut)
    (samplerate, wavedata) = wavfile.read(file)
    s1= spectral_centroid(wavedata,win,samplerate)
    sr1= spectral_rolloff(wavedata,win,samplerate)
    sf1= spectral_flux(wavedata,win,samplerate)
    rms= root_mean_square(wavedata, win, samplerate)
    rms= rms[~numpy.isnan(rms)] #rms array contains NAN values and we have to remove these values
    zcr= zero_crossing_rate(wavedata, win, samplerate)
    MFCC_res=mfcc(wavedata)
    eng= stEnergy(wavedata)
    return s1,sr1,sf1,rms,zcr,MFCC_res,eng,intClass

def zero_crossing_rate(wavedata, block_length, samplerate):
    import numpy as numpy
    # how many blocks have to be processed?
    num_blocks = int(numpy.ceil(len(wavedata)/block_length))

    # when do these blocks begin (time in seconds)?
    timestamps = (numpy.arange(0,num_blocks - 1) * (block_length / float(samplerate)))

    zcr = []

    for i in range(0,num_blocks-1):

        start = i * block_length
        stop  = numpy.min([(start + block_length - 1), len(wavedata)])

        zc = 0.5 * numpy.mean(numpy.abs(numpy.diff(numpy.sign(wavedata[start:stop]))))
        zcr.append(zc)

    return numpy.asarray(zcr) #, numpy.asarray(timestamps)
def spectral_rolloff(wavedata, window_size, sample_rate, k=0.85):
    import numpy as numpy
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    power_spectrum     = numpy.abs(magnitude_spectrum)**2
    timebins, freqbins = numpy.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    timestamps = (numpy.arange(0,timebins - 1) * (timebins / float(sample_rate)))

    sr = []

    spectralSum    = numpy.sum(power_spectrum, axis=1)

    for t in range(timebins-1):

        # find frequency-bin indeces where the cummulative sum of all bins is higher
        # than k-percent of the sum of all bins. Lowest index = Rolloff
        sr_t = numpy.where(numpy.cumsum(power_spectrum[t,:]) >= k * spectralSum[t])[0][0]

        sr.append(sr_t)

    sr = numpy.asarray(sr).astype(float)

    # convert frequency-bin index to frequency in Hz
    sr = (sr / freqbins) * (sample_rate / 2.0)

    return sr #, numpy.asarray(timestamps)


def spectral_flux(wavedata, window_size, sample_rate):
    import numpy as numpy
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    timebins, freqbins = numpy.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    timestamps = (numpy.arange(0,timebins - 1) * (timebins / float(sample_rate)))

    sf = numpy.sqrt(numpy.sum(numpy.diff(numpy.abs(magnitude_spectrum))**2, axis=1)) / freqbins

    return sf[1:] #, numpy.asarray(timestamps)

def spectral_centroid(wavedata, window_size, samplerate):
    import numpy as numpy
    magnitude_spectrum = stft(wavedata, window_size)

    timebins, freqbins = numpy.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    timestamps = (numpy.arange(0,timebins - 1) * (timebins / float(samplerate)))

    sc = []

    for t in range(timebins-1):

        power_spectrum = numpy.abs(magnitude_spectrum[t])**2

        sc_t = numpy.sum(power_spectrum * numpy.arange(1,freqbins+1)) / numpy.sum(power_spectrum)

        sc.append(sc_t)

    sc = numpy.asarray(sc)
    sc = numpy.nan_to_num(sc)

    return sc #, numpy.asarray(timestamps)


def root_mean_square(wavedata, block_length, sample_rate):
    import numpy as numpy
    # how many blocks have to be processed?
    length =len(wavedata)

    num_blocks = int(numpy.ceil(length/block_length))

    #num_blocks= int(num_blocks)
    # when do these blocks begin (time in seconds)?
    timestamps = (numpy.arange(0,num_blocks - 1) * (block_length / float(sample_rate)))


    rms = []
    for i in range(0,num_blocks-1):

        start = i * block_length
        stop  = numpy.min([(start + block_length - 1), len(wavedata)])

        rms_seg = numpy.sqrt(numpy.mean(wavedata[start:stop]**2))

        rms.append(rms_seg)

    return numpy.asarray(rms) #, numpy.asarray(timestamps)

def stft(sig, frameSize, overlapFac=0.5, window=numpy.hanning):
    import numpy as numpy
    from numpy.lib import stride_tricks
    win = window(frameSize)
    hopSize = int(frameSize - numpy.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = numpy.append(numpy.zeros(numpy.floor(frameSize/2.0)), sig)
    # cols for windowing
    cols = numpy.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = numpy.append(samples, numpy.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return numpy.fft.rfft(frames)

def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))
"""
EXTrA COdE:

"""

def joinCSVs(auList,pathLength):
    from collections import OrderedDict
    with open('/home/ubantu/Vikas/Data(D)/STUDY/PRJT/GTZAN Dataset/MFCCfeatureSet.csv', 'w') as csvfile:
        fieldnames = ['fileIndex',\
                    'MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11','MFCC12','MFCC13',\
                    'CLASS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for file in auList:
            string1= file
            cut=string1[pathLength:]
            cut=cut[:-10]
            intClass=ClassToInt(cut)
            (samplerate, wavedata) = wavfile.read(file)
            (MFCCs, mspec, spec) = mfcc(wavedata)
            MFCC_coef=list()
            for ind in range(len(MFCCs)):
                MFCCs[ind][MFCCs[ind] == -inf] = 0
            ran=MFCCs.shape
            ran1=ran[0]
            for ind1 in range(13):
                sum=0
                flag=False
                for ind in range(ran1):
                    sum+=MFCCs[ind,ind1]
                MFCC_coef.append(sum/ran1)
            writer.writerow({'fileIndex':file,\
                            'MFCC1':MFCC_coef[0],'MFCC2':MFCC_coef[1],'MFCC3':MFCC_coef[2],'MFCC4':MFCC_coef[3],\
                            'MFCC5':MFCC_coef[4],'MFCC6':MFCC_coef[5],'MFCC7':MFCC_coef[6],'MFCC8':MFCC_coef[7],\
                            'MFCC9':MFCC_coef[8],'MFCC10':MFCC_coef[9],'MFCC11':MFCC_coef[10],'MFCC12':MFCC_coef[11],\
                            'MFCC13':MFCC_coef[12],'CLASS':intClass})
    with open(r'/home/ubantu/Vikas/Data(D)/STUDY/PRJT/GTZAN Dataset/Pre-featureSet.csv') as f:
        r = csv.reader(f, delimiter=',')
        r.next()
        dict1 = {}
        for row in r:
            dict1.update({row[0]: row[1:]})

    with open(r'/home/ubantu/Vikas/Data(D)/STUDY/PRJT/GTZAN Dataset/MFCCfeatureSet.csv') as f:
        r = csv.reader(f, delimiter=',')
        r.next()
        dict2 = {}
        for row in r:
            dict2.update({row[0]: row[1:]})

    result = OrderedDict()
    for d in (dict1, dict2):
        for key, value in d.iteritems():
            result.setdefault(key, []).extend(value)

    with open(r'/home/ubantu/Vikas/Data(D)/STUDY/PRJT/GTZAN Dataset/TwoClassfeatureSet.csv', 'wb') as f:
        w = csv.writer(f)
        for key, value in result.iteritems():
            w.writerow(value)


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
    #------------------
    # Compute the MFCC
    #------------------
    extract = preemp(input, prefac)
    framed = segment_axis(extract, nwin, over) * w

    # Compute the spectrum magnitude
    spec = numpy.abs(fft(framed, nfft, axis=-1))
    # Filter the spectrum through the triangle filterbank
    arr= numpy.dot(spec,fbank.T)                                         ##CHANGED CODE
   # print "LOG ARRAy =",arr
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
