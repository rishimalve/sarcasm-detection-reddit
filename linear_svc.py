#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:09:47 2019

@author: pranjalihugay
"""

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession, Row
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, FloatType
from pyspark.sql.functions import monotonically_increasing_id, abs,sum
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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

features_df = spark.read.csv("./features.csv",header=True,inferSchema = True)
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


svc = LinearSVC(maxIter=10, regParam=0.1)
df_list = [contrast_based_df,emotion_based_df,text_expression_based_df]
RMSEs = []
MAEs = []
FS = []
Accuracies = []
Precisions = []
Recalls = []
for item in range(len(df_list)):
    print("-----------------RDD: "+str(item)+" -----------------------")
    for i in range(1, 6):
        print("---------------------FOLD "+str(i)+"-----------------------------")
        train, test = df_list[item].randomSplit([0.8, 0.2])
        svcModel = svc.fit(train)
        preds = svcModel.transform(test)
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(preds.select(['label','prediction']))
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="f1")
        f1 = evaluator.evaluate(preds.select(['label','prediction']))
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedRecall")
        Recall = evaluator.evaluate(preds.select(['label','prediction']))
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedPrecision")
        Precision = evaluator.evaluate(preds.select(['label','prediction']))
        FS.append(f1)
        Accuracies.append(accuracy)
        Recalls.append(Recall)
        Precisions.append(Precision)
        