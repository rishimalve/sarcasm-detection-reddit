#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession, Row
from pyspark import SQLContext
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import SparseVector

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

features_rdd = df.rdd

features_rdd = features_rdd.map(lambda line: LabeledPoint(line[0],line[1:]))

MLUtils.saveAsLibSVMFile(features_rdd, "data/input/")

vectorAssembler = VectorAssembler(inputCols=['Exclamations', 'Question_Marks', 'Ellipsis', 'Interjections', 'Capitals', 'RepeatLetters', 'Positive_word_count', 'Negative_word_count',
                                             'Polarity_flips', 'Nouns', 'Verbs', 'PositiveIntensifiers', 'NegativeIntensifiers', 'Bigrams', 'Trigram', 'Passive_aggressive_count'], outputCol='features')

v_df = vectorAssembler.transform(df)
# v_df = v_df.withColumn('label', col("Trip_Duration"))
v_df = v_df.select(['label','features'])

v_train, v_test = v_df.randomSplit([0.7, 0.3])



final_rdd = v_df.rdd.map(lambda row: LabeledPoint(
    row['label'], row['features'].toArray()))

data = spark.read.format("libsvm").load("/Users/rishimalve/Documents/Masters/data.txt")

train, test = data.randomSplit([0.7, 0.3])

train_df = sqlContext.createDataFrame(train, ['label', 'features'])

layers = [16, 5, 4, 2]

trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

model = trainer.fit(train)
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")