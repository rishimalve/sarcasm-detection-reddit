#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:08:16 2019

@author: pranjalihugay
"""

# =============================================================================
# Naive Bayes
# Logistic Regression
# Support Vector Machines
# Random Forest
# Neural Networks
# Decision Trees
# =============================================================================
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession, Row
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, FloatType
from pyspark.sql.functions import monotonically_increasing_id, abs,sum
from pyspark.mllib.evaluation import MulticlassMetrics
import math
import statistics
import pyspark.sql.functions as F


#Create a spark session and name it
spark = SparkSession.builder.appName('phugay_app1%s').getOrCreate()
#Create a spark context in session
sc = spark.sparkContext
#Reduce output by only showing me the errors
sc.setLogLevel("ERROR")
#SQL Context
sqlContext = SQLContext(sc)

features_df = spark.read.csv("./data/features.csv",header=True,inferSchema = True)
contrast_based_features = features_df.select(["label","Polarity_flips","Positive_word_count","Negative_word_count"])
emotion_based_features = features_df.select(["label","PositiveIntensifiers","NegativeIntensifiers","RepeatLetters","Bigrams","Trigram"])
text_expression_based_features = features_df.select(["label","Nouns","Verbs","Exclamations","Question_Marks","Interjections","Ellipsis","Capitals","Passive_aggressive_count"])

cb_vectorAssembler = VectorAssembler(inputCols = ["Polarity_flips","Positive_word_count",
                                                  "Negative_word_count"], outputCol = 'features')
eb_vectorAssembler = VectorAssembler(inputCols = ["PositiveIntensifiers","NegativeIntensifiers",
                                                  "RepeatLetters","Bigrams","Trigram"],outputCol = 'features')
teb_vectorAssembler = VectorAssembler(inputCols = ["Nouns","Verbs","Exclamations","Question_Marks",
                                                   "Interjections","Ellipsis","Capitals",
                                                   "Passive_aggressive_count"],outputCol = 'features')

contrast_based_transdf = cb_vectorAssembler.transform(contrast_based_features)
contrast_based_df = contrast_based_transdf.select(["features","label"])
emotion_based_transdf = eb_vectorAssembler.transform(emotion_based_features)
emotion_based_df = emotion_based_transdf.select(["features","label"])
text_expression_transdf = teb_vectorAssembler.transform(text_expression_based_features)
text_expression_based_df = text_expression_transdf.select(["features","label"])
contrast_based_rdd = contrast_based_df.rdd.map(lambda row: LabeledPoint(row['label'], row['features'].toArray()))
emotion_based_rdd = emotion_based_df.rdd.map(lambda row: LabeledPoint(row['label'], row['features'].toArray()))
text_expression_based_rdd = text_expression_based_df.rdd.map(lambda row: LabeledPoint(row['label'], row['features'].toArray()))

rdd_list = [contrast_based_rdd,emotion_based_rdd,text_expression_based_rdd]
RMSEs = []
MAEs = []
FS = []
Accuracies = []
Precisions = []
Recalls = []
for item in range(len(rdd_list)):
    print("-----------------RDD: "+str(item)+" -----------------------")
    for i in range(1, 6):
        print("---------------------FOLD "+str(i)+"-----------------------------")
        features_train_df, features_test_df = rdd_list[item].randomSplit([0.8, 0.2])    
        testcount = features_test_df.count()    
        model = DecisionTree.trainRegressor(features_train_df, categoricalFeaturesInfo={},impurity='variance', maxDepth=5, maxBins=32)
        predictions = model.predict(features_test_df.map(lambda x: (x.features,)))
        testpredictions = predictions.map(lambda pr: (pr,))
        testlabels = features_test_df.map(lambda lp: (lp.label,))    
        label_schema = StructType([StructField('Labels', DoubleType(), True)])
        predict_schema = StructType([StructField('Predictions', DoubleType(), True)])
        result_schema = StructType([StructField('Labels', DoubleType(), True),StructField('Predictions', DoubleType(), True)])    
        label_df = sqlContext.createDataFrame(testlabels, label_schema)
        predict_df = sqlContext.createDataFrame(testpredictions, predict_schema)    
        ddf1 = label_df.withColumn("row_id", monotonically_increasing_id())
        ddf2 = predict_df.withColumn("row_id", monotonically_increasing_id())   
        result = ddf1.join(ddf2, (ddf1.row_id == ddf2.row_id)).drop("row_id")
        result_diffsqr = result.withColumn("Diffsqr",(result.Labels - result.Predictions)**2)
        result_absdiff = result_diffsqr.withColumn("Absdiff",abs(result.Labels - result.Predictions))    
        total = result_absdiff.select(F.sum("Diffsqr")).collect()[0][0]
        mae_total = result_absdiff.select(F.sum("Absdiff")).collect()[0][0]
        MSE = total/testcount
        MAE = mae_total/testcount
        RMSE = math.sqrt(MSE)
        print("rmse",RMSE)
        print("mse",MSE)
        print("mae",MAE)    
        predictionAndLabels=result.rdd.map(lambda x:(x[1],x[0]))
        metrics = MulticlassMetrics(predictionAndLabels)
        cm=metrics.confusionMatrix().toArray()
        f1Score = metrics.fMeasure()
        accuracy=(cm[0][0]+cm[1][1])/cm.sum()
        recall = metrics.recall()
        precision = metrics.precision()
        print("f1Score",f1Score)
        print("accuracy",accuracy)
        print("precision",precision)
        print("recall",recall)
        RMSEs.append(RMSE)
        MAEs.append(MAE)
        Accuracies.append(accuracy)
        Precisions.append(precision)
        Recalls.append(recall)
        FS.append(f1Score)
        
print('Root Mean Squared Error = ' + str(statistics.mean(RMSEs)))
print('Mean Absolute Error = ' + str(statistics.mean(MAEs)))