from SpectCentroid import spectral_centroid
from SpectRollOff import spectral_rolloff
from STFT import stft
import warnings
from RootMeanSquare import root_mean_square
from SpectFlux import spectral_flux
#import audiotools as aT
from music.feature.Classification import classification_trainTest, readCSVFile, classification_SVMtest
from music.feature.FeatureExtraction import featureExtraction
from music.feature.MFCC import stEnergy
from music.feature.ParallelFeatExt import parFeatureExtraction
from music.feature.ZeroCrossingRate import zero_crossing_rate
import time
warnings.filterwarnings('ignore')

def main():
    t0=time.time()
    print "for traing system press: 1 \n for Testing user tracks press: 2 or 3"
    user_input=input()
    if user_input==1:
        print "Enter training dataset location:  "
        dataset_location=raw_input()
        pathLength=len(dataset_location)
        #print pathLength
        print "Extracting Features!!!! Please wait......"
        parFeatureExtraction(dataset_location,pathLength+1)
        #featureExtraction(dataset_location,pathLength+1)
        print "features extracted...!!!"
        print "Running SVM model...."
        (featureSet, classlables)=readCSVFile()                            #read CSV file and retruns feature set and class lables
        classification_trainTest(featureSet,classlables)
    elif user_input==3:
        print "Enter training dataset location:  "
        dataset_location=raw_input()
        pathLength=len(dataset_location)
        print "Extracting Features..."
        parFeatureExtraction(dataset_location,pathLength+1)
        print "features extracted...!!!"
        print "Running SVM Test model...."
        (featureSet, classlables)=readCSVFile()                            #read CSV file and retruns feature set and class lables
        classification_SVMtest(featureSet,classlables)

    else:
        print "Enter test dataset location:  "
        dataset_location=raw_input()
        pathLength=len(dataset_location)
        print pathLength
        print "Extracting Features!!!! Please wait......"
        featureExtraction(dataset_location,pathLength+1)
        print "features extracted...!!!"
        print "Running SVM Test model...."
        (featureSet, classlables)=readCSVFile()                            #read CSV file and retruns feature set and class lables
        classification_SVMtest(featureSet,classlables)
    #print "Done in = ",time.time()-t0

if __name__ == '__main__':
    main()

