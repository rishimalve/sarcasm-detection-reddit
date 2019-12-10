#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession, Row
from pyspark import SQLContext
import re as re
from pyspark.ml.feature import CountVectorizer , IDF
from pyspark.mllib.linalg import Vector, Vectors, SparseVector
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.clustering import LDA
import numpy as np
from pyspark.sql.functions import monotonically_increasing_id, row_number, col, explode, size
from pyspark.sql import Window
import pyLDAvis

# Create a spark session and name it.
spark = SparkSession.builder.appName('rishi-malve').getOrCreate()

# Create a spark context in session.
sc = spark.sparkContext

sc.setLogLevel("ERROR")

# SQL Context.
sqlContext = SQLContext(sc)

df = spark.read.csv('data/input_TM.csv', header=True, inferSchema=True)
df.printSchema()
df.show()

comments = df.rdd
comments = comments.map(lambda x: x['combined'].encode("utf-8")).filter(lambda x: x is not None)

comments = comments.map(lambda document: re.split(" ", document)).map(
    lambda word: [x for x in word if (len(x) > 3) and x != '']).zipWithIndex()

df_comments = sqlContext.createDataFrame(comments, ["list_of_words",'index'])

# TF
cv = CountVectorizer(inputCol="list_of_words", outputCol="raw_features", vocabSize=50000, minDF=10.0)
cvmodel = cv.fit(df_comments)
result_cv = cvmodel.transform(df_comments)
# IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(result_cv)
result_tfidf = idfModel.transform(result_cv)

lda = LDA(k=5, maxIter=100)
model = lda.fit(result_tfidf[['index','features']])

transformed = model.transform(result_tfidf)
# transformed.show(truncate=False)
model.describeTopics(8).show()

# ll = model.logLikelihood(result_tfidf[['index','features']])
# lp = model.logPerplexity(result_tfidf[['index','features']])

vocabulary = {}
j = 0
for i in cvmodel.vocabulary:
    vocabulary[j] = i.encode("utf-8")
    j += 1

model_df = model.describeTopics(8)

termIndices_rdd = model.describeTopics(8).select('termIndices').rdd.map(lambda x: x[0])

termIndices_rdd = termIndices_rdd.map(lambda x: [vocabulary[y] for y in x]).zipWithIndex()

termIndices_df = sqlContext.createDataFrame(termIndices_rdd, ['term_words', 'index'])

final_df = model_df.join(termIndices_df, (model_df.topic == termIndices_df.index)).drop('index')

final_df.select('topic','term_words').show(truncate = False)

ll = termIndices_rdd.map(lambda x: x[0]).collect()

topicDistribution_rdd = transformed.select('topicDistribution').rdd.map(lambda x: list(x[0].values)).map(lambda x: x.index(max(x))).zipWithIndex()

topicDistribution_df = sqlContext.createDataFrame(topicDistribution_rdd, ['topic', 'index_1'])

transformed_df = transformed.join(topicDistribution_df, (transformed.index == topicDistribution_df.index_1)).drop(
    'index_1', 'raw_features', 'features').orderBy('index')


df = df.join(transformed_df, (df._c0 == transformed_df.index)).orderBy('_c0').drop('index','_c0')

df = df.drop('list_of_words', 'topicDistribution', 'external_author_id', 'new_june_2018')

df.write.format("csv").save('/Users/rishimalve/Documents/Masters/Sem-3/CS 657/HW3/data/topics')


with open('/Users/rishimalve/Documents/Masters/Sem-3/CS 657/HW3/data/list.csv', 'w') as f:
    ll=[str(x) for x in ll]
    for item in ll:
        f.write("%s\n" % item)

def format_data_to_pyldavis(df_filtered, count_vectorizer, transformed, lda_model):
    xxx = df_filtered.select((explode(df_filtered.list_of_words)).alias("words")).groupby("words").count()
    word_counts = {r['words']:r['count'] for r in xxx.collect()}
    word_counts = [word_counts[w] for w in count_vectorizer.vocabulary]


    data = {'topic_term_dists': np.array(lda_model.topicsMatrix().toArray()).T, 
            'doc_topic_dists': np.array([x.toArray() for x in transformed.select(["topicDistribution"]).toPandas()['topicDistribution']]),
            'doc_lengths': [r[0] for r in df_filtered.select(size(df_filtered.list_of_words)).collect()],
            'vocab': count_vectorizer.vocabulary,
            'term_frequency': word_counts}

    return data

def filter_bad_docs(data):
    bad = 0
    doc_topic_dists_filtrado = []
    doc_lengths_filtrado = []

    for x,y in zip(data['doc_topic_dists'], data['doc_lengths']):
        if np.sum(x)==0:
            bad+=1
        elif np.sum(x) != 1:
            bad+=1
        elif np.isnan(x).any():
            bad+=1
        else:
            doc_topic_dists_filtrado.append(x)
            doc_lengths_filtrado.append(y)

    data['doc_topic_dists'] = doc_topic_dists_filtrado
    data['doc_lengths'] = doc_lengths_filtrado



# FORMAT DATA AND PASS IT TO PYLDAVIS
data = format_data_to_pyldavis(df_tweets, cvmodel, transformed, model)
filter_bad_docs(data) # this is, because for some reason some docs apears with 0 value in all the vectors, or the norm is not 1, so I filter those docs.
py_lda_prepared_data = pyLDAvis.prepare(**data)
pyLDAvis.save_html(py_lda_prepared_data, '/Users/rishimalve/Documents/Masters/Sem-3/CS 657/HW3/clusters.html')