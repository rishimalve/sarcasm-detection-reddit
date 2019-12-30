#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark.ml.classification import NaiveBayes
from pyspark.sql import SparkSession, Row
from pyspark import SQLContext
from pyspark.ml.classification import NaiveBayes
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


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

contrast_based_features = df.select(["label","Polarity_flips","Positive_word_count","Negative_word_count"])
emotion_based_features = df.select(["label","PositiveIntensifiers","NegativeIntensifiers","RepeatLetters"])
text_expression_based_features = df.select(["label","Nouns","Verbs","Exclamations","Question_Marks","Interjections","Ellipsis","Capitals","Passive_aggressive_count"])

cb_vectorAssembler = VectorAssembler(inputCols = ["Polarity_flips","Positive_word_count",
                                                  "Negative_word_count"], outputCol = 'features')
eb_vectorAssembler = VectorAssembler(inputCols = ["PositiveIntensifiers","NegativeIntensifiers",
                                                  "RepeatLetters"],outputCol = 'features')
teb_vectorAssembler = VectorAssembler(inputCols = ["Nouns","Verbs","Exclamations","Question_Marks",
                                                   "Interjections","Ellipsis","Capitals",
                                                   "Passive_aggressive_count"],outputCol = 'features')

contrast_based_transdf = cb_vectorAssembler.transform(contrast_based_features)
contrast_based_df = contrast_based_transdf.select(["features","label"])
emotion_based_transdf = eb_vectorAssembler.transform(emotion_based_features)
emotion_based_df = emotion_based_transdf.select(["features","label"])
text_expression_transdf = teb_vectorAssembler.transform(text_expression_based_features)
text_expression_based_df = text_expression_transdf.select(["features","label"])

nb = NaiveBayes(smoothing=1)

for i in range(1,6):

    print("Fold Number : " + str(i))
    print("############## contrast based ##############")

    train_rdd, test_rdd = contrast_based_df.randomSplit([0.8, 0.2])

    model = nb.fit(train_rdd)

    predictions = model.transform(test_rdd)
    
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictions.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedRecall")
    weightedRecall = evaluator.evaluate(predictions.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedPrecision")
    weightedPrecision = evaluator.evaluate(predictions.select(['label','prediction']))
    
    print("f1 : " + str(f1))
    print("accuracy : " + str(accuracy))
    print("weightedRecall : " + str(weightedRecall))
    print("weightedPrecision : " + str(weightedPrecision))


    print("############## emotion based ##############")

    train_rdd, test_rdd = emotion_based_transdf.randomSplit([0.8, 0.2])

    model = nb.fit(train_rdd)

    predictions = model.transform(test_rdd)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictions.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedRecall")
    weightedRecall = evaluator.evaluate(predictions.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedPrecision")
    weightedPrecision = evaluator.evaluate(predictions.select(['label','prediction']))

    print("f1 : " + str(f1))
    print("accuracy : " + str(accuracy))
    print("weightedRecall : " + str(weightedRecall))
    print("weightedPrecision : " + str(weightedPrecision))


    print("############## text expression based ##############")

    train_rdd, test_rdd = text_expression_transdf.randomSplit([0.8, 0.2])

    model = nb.fit(train_rdd)

    predictions = model.transform(test_rdd)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictions.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedRecall")
    weightedRecall = evaluator.evaluate(predictions.select(['label','prediction']))
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName="weightedPrecision")
    weightedPrecision = evaluator.evaluate(predictions.select(['label','prediction']))

    print("f1 : " + str(f1))
    print("accuracy : " + str(accuracy))
    print("weightedRecall : " + str(weightedRecall))
    print("weightedPrecision : " + str(weightedPrecision))
