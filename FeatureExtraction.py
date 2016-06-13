from SpectCentroid import spectral_centroid
from SpectRollOff import spectral_rolloff
import time
import csv
import warnings
from Tkinter import *
from RootMeanSquare import root_mean_square
from SpectFlux import spectral_flux
from music.feature.MFCC import stEnergy
from music.feature.ZeroCrossingRate import zero_crossing_rate
warnings.filterwarnings('ignore')

# numerical processing and scientific libraries
import numpy as np
from numpy import inf
# signal processing
from scipy.io                     import wavfile
from scikits.talkbox.features     import mfcc
import glob


def featureExtraction(dataset_location,pathLength):
    t0=time.time()
    path =dataset_location+'/*.wav'
    print path
    files=glob.glob(path)
    auList=list()
    for file in files:
        auList.append(file)

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


    with open('/home/ubantu/TwoClassfeatureSet.csv', 'w') as csvfile:
        fieldnames = ['Spect Centroid', 'Spect Rolloff','Spect Flux','RMS','ZCR','SC_SD','SR_SD','SF_SD','ZCR_SD','energy',\
                      'MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11','MFCC12','MFCC13',\
                      'CLASS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        #auList=auList[0:100]
        for file in auList:
            string1= file
            cut=string1[pathLength:]
            cut=cut[:-10]
            intClass=ClassToInt(cut)
            (samplerate, wavedata) = wavfile.read(file)
            s1= spectral_centroid(wavedata,512,samplerate)
            sr1= spectral_rolloff(wavedata,512,samplerate)

            sf1= spectral_flux(wavedata,512,samplerate)
            rms= root_mean_square(wavedata, 512, samplerate)
            rms= rms[~np.isnan(rms)] #rms array contains NAN values and we have to remove these values
            zcr= zero_crossing_rate(wavedata, 512, samplerate)
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
            #print ind1," MFCC_coef====",MFCC_coef
            eng= stEnergy(wavedata)
            writer.writerow({'Spect Centroid':s1.mean().astype(float), 'Spect Rolloff':sr1.mean().astype(float),'Spect Flux':sf1.mean().astype(float),'RMS':rms.mean().astype(float),'ZCR':zcr.mean().astype(float),\
                             'SC_SD':s1.std().astype(float),'SR_SD':sr1.std().astype(float),'SF_SD':sf1.std().astype(float),'ZCR_SD':zcr.std().astype(float),'energy':eng.astype(float),\
                             'MFCC1':MFCC_coef[0].astype(float),'MFCC2':MFCC_coef[1].astype(float),'MFCC3':MFCC_coef[2].astype(float),'MFCC4':MFCC_coef[3].astype(float),\
                             'MFCC5':MFCC_coef[4].astype(float),'MFCC6':MFCC_coef[5].astype(float),'MFCC7':MFCC_coef[6].astype(float),'MFCC8':MFCC_coef[7].astype(float),\
                             'MFCC9':MFCC_coef[8].astype(float),'MFCC10':MFCC_coef[9].astype(float),'MFCC11':MFCC_coef[10].astype(float),'MFCC12':MFCC_coef[11].astype(float),\
                             'MFCC13':MFCC_coef[12].astype(float),'CLASS':intClass})
    tEnd =time.time()-t0
    print "feature extraction done in=   ",tEnd

