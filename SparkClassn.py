

                ## Classification on Spark: Spark MLLib package supports binary SVM only

import os
import sys
import time
import time

os.environ['SPARK_HOME']="/home/ubantu/spark"
import time


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
    from pyspark.mllib.classification import SVMWithSGD, SVMModel
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import DecisionTreeClassifier
    from pyspark.ml.feature import StringIndexer, VectorIndexer
    from pyspark.ml.classification import MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print "Can not import Spark Modules", e
    sys.exit(1)

sc = SparkContext( 'local', 'pyspark')
sqlContext = SQLContext(sc)
t0=time.time()
print "========SPARK SVM========="

#def parsePoint(line):
#    values = [float(x) for x in line.split(' ')]
#    return LabeledPoint(values[0], values[1:])

def parsePoint1(line):
    keys  = [float(x) for x in line.split(",")]
    return LabeledPoint(keys[0],keys[1:])

#scdata = sc.textFile("/home/ubantu/TwoClassfeatureSet1.txt")
#parsedData = scdata.map(parsePoint)

scdata1 = sc.textFile("/home/ubantu/TwoClassfeatureSet.csv")

parsedData1= scdata1.map(parsePoint1)

training, test = parsedData1.randomSplit([0.8, 0.2], seed=0)
#print parsedData


# Build the model
model = SVMWithSGD.train(training, iterations=100)

#Evaluating the model on training data
labelsAndPreds = training.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(training.count())
labelsAndPreds = test.map(lambda p: (p.label, model.predict(p.features)))
trainErr1 = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test.count())
print "=============================================================================================================================="
print "Training Accuracy = " , (1-float(trainErr))*100,"%"
print "Testing Accuracy = " , (1-float(trainErr1))*100,"%"


def naiveBayeseian():

    def parseLine(line):
        keys  = [float(x) for x in line.split(",")]
        #return LabeledPoint(keys[0],keys[1:])
        return keys
    scdata1 = sc.textFile("/home/ubantu/TwoClassfeatureSet.csv")
    data= scdata1.map(parseLine)
    splits = data.randomSplit([0.8, 0.2], 1234)
    train = splits[0]
    test = splits[1]
    layers = [30, 20, 20, 2]
    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
    # train the model
    model = trainer.fit(train)
    # compute precision on the test set
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="precision")
    print("Precision:" + str(evaluator.evaluate(predictionAndLabels)))
    #print test.take(5)



#naiveBayeseian()

print "++++++++++++++++++++++++++++++++++++++++++++++++"
print "Done in = ",time.time()-t0

