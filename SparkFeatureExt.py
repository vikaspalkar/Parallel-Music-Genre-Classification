
              ## incomplete file: Spark feature extraction code yet to finish
import wave
import os
import sys
from SpectCentroid import spectral_centroid
from SpectRollOff import spectral_rolloff
from STFT import stft
import csv
import warnings
from RootMeanSquare import root_mean_square
from SpectFlux import spectral_flux
from music.feature.MFCC import stEnergy
from music.feature.ZeroCrossingRate import zero_crossing_rate
warnings.filterwarnings('ignore')

# numerical processing and scientific libraries
import numpy as np
import scipy
from pprint import pprint
# signal processing
from scipy.io                     import wavfile

from scikits.talkbox.features     import mfcc
import time
os.environ['SPARK_HOME']="/home/ubantu/spark"


# Append pyspark  to Python Path
sys.path.append("/home/ubantu/spark/python/")
sys.path.append("/home/ubantu/spark/python/lib/py4j-0.9-src.zip")
#sys.path.append("/home/ubantu/spark-1.6.0/python/build/")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SQLContext,Row
    from pyspark.sql.types import StructType,StructField,StringType,FloatType
    from pyspark.mllib.linalg import Vectors

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print "Can not import Spark Modules", e
    sys.exit(1)



sc = SparkContext( 'local', 'pyspark')
sqlContext = SQLContext(sc)

t0=time.time()

import glob
path ='/home/ubantu/Vikas/Data(D)/STUDY/PRJT/GTZAN Dataset/TwoclassDataset1/*.wav'
files=glob.glob(path)
auList=list()

f=open('/home/ubantu/Vikas/Data(D)/STUDY/PRJT/GTZAN Dataset/TwoClassfeatureTEXT.txt', 'w')

for file in files:
    auList.append(file)
    f.write(file+"\n")
#print auList

f.close()
scAuList=sc.parallelize(auList)
print scAuList.collect()
def ClassToInt(name):
    #values = name[69:]                   ##Spark Call
    #name=values[:-10]
    #name=str(name)
    #print name
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
#print scAuList                             ##Spark Call

intNo= scAuList.map(ClassToInt)
headRowa=intNo.take(10)
print  intNo.collect()
#lst=list()
#for file in range(10):
#    print auList[file], "Class== "
#    pprint(headRowa[file])

def sparkFeatureExt(line):
        print line
        string1= file
        cut=string1[72:]
        cut=cut[:-10]
        intClass=ClassToInt(cut)

        (samplerate, wavedata) = wavfile.read(file)
        (s1,n1)= spectral_centroid(wavedata,512,samplerate)
        (sr1,nr1)= spectral_rolloff(wavedata,512,samplerate)
        (sf1,nf1)= spectral_flux(wavedata,512,samplerate)
        (rms,ts) = root_mean_square(wavedata, 512, samplerate);
        rms= rms[~np.isnan(rms)] #rms array contains NAN values and we have to remove these values
        (zcr,ts1) = zero_crossing_rate(wavedata, 512, samplerate);
        (MFCCs, mspec, spec) = mfcc(wavedata)
        MFCC_coef=list()
        ran=MFCCs.shape
        ran1=ran[0]
        for ind1 in range(13):
            sum=0
            summ=0
            for ind in range(ran1):
                sum+=MFCCs[ind,ind1]
            MFCC_coef.append(sum/ran1)
        eng= stEnergy(wavedata)
        #Win = 0.050
        #Step = 0.050
        #eps = 0.00000001
        return s1,sr1,sf1,rms,zcr,eng,MFCC_coef,intClass
        #f.write(intClass,' ',s1.mean().astype(float), ' ',sr1.mean().astype(float),' ',sf1.mean().astype(float),' ',rms.mean().astype(float),' ',zcr.mean().astype(float),' ',eng.astype(float),' ',MFCC_coef[0].astype(float),' ',MFCC_coef[1].astype(float),' ',MFCC_coef[2].astype(float),' ',MFCC_coef[3].astype(float),' ',MFCC_coef[4].astype(float),' ',MFCC_coef[5].astype(float),' ',MFCC_coef[6].astype(float),' ',MFCC_coef[7].astype(float),' ',MFCC_coef[8].astype(float),' ',MFCC_coef[9].astype(float),' ',MFCC_coef[10].astype(float),' ',MFCC_coef[11].astype(float),' ',MFCC_coef[12].astype(float))
    #f.close()
"""
with open('/home/ubantu/Vikas/Data(D)/STUDY/PRJT/GTZAN Dataset/TwoClassfeatureSet.csv', 'w') as csvfile:
        fieldnames = ['Spect Centroid', 'Spect Rolloff','Spect Flux','RMS','ZCR','energy','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11','MFCC12','MFCC13','CLASS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        scdata1 = sc.textFile('/home/ubantu/Vikas/Data(D)/STUDY/PRJT/GTZAN Dataset/TwoClassfeatureTEXT.txt')
        data= scdata1.map(sparkFeatureExt)
        print data.take(5)
        #feature=scAuList.map(sparkFeatureExt)
        #writer.writerow({'Spect Centroid':s1.mean().astype(float), 'Spect Rolloff':sr1.mean().astype(float),'Spect Flux':sf1.mean().astype(float),'RMS':rms.mean().astype(float),'ZCR':zcr.mean().astype(float),'energy':eng.astype(float),'MFCC1':MFCC_coef[0].astype(float),'MFCC2':MFCC_coef[1].astype(float),'MFCC3':MFCC_coef[2].astype(float),'MFCC4':MFCC_coef[3].astype(float),'MFCC5':MFCC_coef[4].astype(float),'MFCC6':MFCC_coef[5].astype(float),'MFCC7':MFCC_coef[6].astype(float),'MFCC8':MFCC_coef[7].astype(float),'MFCC9':MFCC_coef[8].astype(float),'MFCC10':MFCC_coef[9].astype(float),'MFCC11':MFCC_coef[10].astype(float),'MFCC12':MFCC_coef[11].astype(float),'MFCC13':MFCC_coef[12].astype(float),'CLASS':intClass})
        csvfile.close()
"""
print "Done in = ",time.time()-t0