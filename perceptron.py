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

contrast_based = spark.read.format("libsvm").load("/Users/rishimalve/Documents/Masters/Sem-3/CS_657/final_project/data/contrast_based.txt")

text_expression_based = spark.read.format("libsvm").load("/Users/rishimalve/Documents/Masters/Sem-3/CS_657/final_project/data/text_expression_based.txt")

emotion_based = spark.read.format("libsvm").load("/Users/rishimalve/Documents/Masters/Sem-3/CS_657/final_project/data/emotion_based.txt")

for i in range(1,6):

    print("Fold Number : " + str(i))
    print("############## contrast based ##############")

    train, test = contrast_based.randomSplit([0.8, 0.2])

    layers = [3, 5, 5, 2]

    trainer = MultilayerPerceptronClassifier(maxIter=5000, layers=layers)

    model = trainer.fit(train)
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedRecall")
    weightedRecall = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedPrecision")
    weightedPrecision = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))

    print("f1 : " + str(f1))
    print("accuracy : " + str(accuracy))
    print("weightedRecall : " + str(weightedRecall))
    print("weightedPrecision : " + str(weightedPrecision))


    print("############## text expression based ##############")

    train, test = text_expression_based.randomSplit([0.8, 0.2])

    layers = [8, 14, 14, 2]

    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers)

    model = trainer.fit(train)
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedRecall")
    weightedRecall = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedPrecision")
    weightedPrecision = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))

    print("f1 : " + str(f1))
    print("accuracy : " + str(accuracy))
    print("weightedRecall : " + str(weightedRecall))
    print("weightedPrecision : " + str(weightedPrecision))


    print("############## emotion based ##############")

    train, test = emotion_based.randomSplit([0.8, 0.2])

    layers = [3, 5, 5, 2]

    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers)

    model = trainer.fit(train)
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedRecall")
    weightedRecall = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedPrecision")
    weightedPrecision = evaluator.evaluate(predictionAndLabels.select(['label','prediction']))

    print("f1 : " + str(f1))
    print("accuracy : " + str(accuracy))
    print("weightedRecall : " + str(weightedRecall))
    print("weightedPrecision : " + str(weightedPrecision))
