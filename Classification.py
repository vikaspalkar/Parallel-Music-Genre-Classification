     #FILE HANDLES CLASSIFICATION TASK


#from music.feature import *
from numpy import inf
import csv
import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import cPickle

def readCSVFile():
    csv_file_object = csv.reader(open('/home/ubantu/TwoClassfeatureSet.csv', 'rb'))
#    csvdata = csv.reader(open("/home/ubantu/BluesTest.csv"))     #Blues Test

    header = csv_file_object.next()  # The next() command just skips the
 #   header1 = csvdata.next()                                  # first line which is a header
    data=[]                          # Create a variable called 'data'.
    dataTemp=[]
    for row in csv_file_object:      # Run through each row in the csv file,
        data.append(row)             # adding each row to the data variable,

    data = np.array(data)
    dummy=data
    dummy= dummy.astype(float)
    fileSize=len(dummy)
    for ind in range(len(dummy)):
        dummy[ind][dummy[ind] == -inf] = 0        #set inf values to zero

    dummy=np.ma.compress_rows(np.ma.fix_invalid(dummy))    # remove nan and inf valued rows

    new_data=dummy                              #numpy array without nan and inf valued rows

    data1= new_data[:,0:23]                   #feature vector
    data1_y=new_data[:,23]                    # class vector
    # normalize the data attributes
    standardized_data1 = preprocessing.scale(data1)
    #normalized_data1 = preprocessing.normalize(data1)
    return standardized_data1,data1_y


def classification_trainTest(standardized_data1,data1_y):

    result=[]

    for i in range(10):         #number of iterations for cross validations

        # split dataset into traing and testing subsets (90% : 10%)
        X_train1, X_test1, y_train1, y_test1 = train_test_split(standardized_data1, data1_y, test_size=0.10)

        res=SVM(standardized_data1, X_test1, data1_y, y_test1)      #classification method

        result.append(res)

        avg_result=sum(result)/10             #average accuracy of system

    print "Final avg accuracy==",avg_result

    classification_SVMtest(X_test1,y_test1)

    t0=time.time()
    print "Classification Done in = ",time.time()-t0

def SVM(X_train1, X_test1, y_train1, y_test1):

    print "====SVM===="
    from sklearn import metrics
    from sklearn.svm import SVC
    # fit a SVM model to the data
    model = SVC()
    model.fit(X_train1,y_train1)
    with open('my_dumped_classifier.pkl', 'wb') as fid:
        cPickle.dump(model, fid)
    #print(model)
    # make predictions
    expected = y_test1
    predicted = model.predict(X_test1)

    # summarize the fit of the model
    print metrics.accuracy_score(expected,predicted)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    return metrics.accuracy_score(expected,predicted)


def classification_SVMtest(X_test1,y_test1):
    print "====SVM====TEST============"
    from sklearn import metrics
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        model_loaded = cPickle.load(fid)

    # make predictions
    expected = y_test1
    predicted = model_loaded.predict(X_test1)

    # summarize the fit of the model
    print "Predicted=> ",predicted
    print "Expected=>  ",expected
    print metrics.accuracy_score(expected,predicted)
    #print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    #return metrics.accuracy_score(expected,predicted)

def logisticRgression(X_train1, X_test1, y_train1, y_test1):
    print "=============Logistic Regression======="
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()

    model.fit(X_train1, y_train1)
    print(model)
    # make predictions
    expected = y_test1
    predicted = model.predict(X_test1)
    # summarize the fit of the model
    #print predicted,"==>>",expected
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

def NaiveBayesian(X_train1, X_test1, y_train1, y_test1):

    print "======NAive Bayesian classification===="
    from sklearn import metrics
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train1, y_train1)
    print(model)
    # make predictions
    expected = y_test1
    predicted = model.predict(X_test1)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


def KNN(X_train1, X_test1, y_train1, y_test1):
    print "========KNN======"

    from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier
    # fit a k-nearest neighbor model to the data
    model = KNeighborsClassifier()
    model.fit(X_train1, y_train1)
    print(model)
    # make predictions
    expected = y_test1
    predicted = model.predict(X_test1)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))




