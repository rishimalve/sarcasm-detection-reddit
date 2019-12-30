#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession, Row
from pyspark import SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint

# Create a spark session and name it.
spark = SparkSession.builder.appName('rishi-malve').getOrCreate()

# Create a spark context in session.
sc = spark.sparkContext

sc.setLogLevel("ERROR")

# SQL Context.
sqlContext = SQLContext(sc)

df = spark.read.csv('data/features.csv', header=True, inferSchema=True)
df.printSchema()
df.show()


vectorAssembler = VectorAssembler(inputCols=['Exclamations', 'Question_Marks', 'Ellipsis', 'Interjections', 'Capitals', 'RepeatLetters', 'Positive_word_count', 'Negative_word_count',
                                             'Polarity_flips', 'Nouns', 'Verbs', 'PositiveIntensifiers', 'NegativeIntensifiers', 'Bigrams', 'Trigram', 'Passive_aggressive_count'], outputCol='features')

v_df = vectorAssembler.transform(df)
v_df = v_df.select(['label','features'])



train_rdd, test_rdd = v_df.randomSplit([0.8, 0.2])

lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=1000)
lrModel = lr.fit(train_rdd)

import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()


predictions = lrModel.transform(test_rdd)
predictions.show(10)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))