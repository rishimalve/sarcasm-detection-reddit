# Sarcasm Detection in Reddit Comments using PySpark
In this project, we are presenting a comparative study of various classification algorithms in machine learning used to classify sarcastic & non-sarcastic tweets with some a variety of features for better accuracy.

## Authors
- [Rishi Malve](https://www.linkedin.com/in/rishi-malve-28b568a4/)
- [Pranjali Hugay](https://www.linkedin.com/in/phugay/)
 
## Classifiers Used
- Naive Bayes
- Linear Support Vector Machine
- Multilayer Perceptron Networks
- Decision Trees Regressor

## Feature Extraction
- We have based all our features solely on the “comment” column of our dataset. Labels were extracted along with these comments. Label = 1 indicates sarcastic comment and Label = 0 otherwise.
- Sentiment analysis was the key to our analysis of these comments. We extracted of the ‘VADER’ package from nltk library and its object called “SentimentIntensityAnalyzer”.
- Lemmatization was done using “WordNetLemmatizer” from nltk library.
- Stemming was done using “PortStemmer” from the nltk library.
- We depended on stop words from the nltk corpus.
- Our approach involved building lists for each of 16 features that we decided to extract from the comments. We decided to consider all of these features keeping in mind that each of these would play a key role in building up sarcasm in comments.
- The 16 features include punctuations like exclamations, question marks and ellipses (...), interjections, Capital words, words with repeated letters, positive words, negative words, polarity flips, nouns, verbs, positive intensifiers, negative intensifiers, bigrams, trigrams
and unigrams.
- For each of these features, we extracted their count in each comment.
- For features like interjections (e.g. argh, ohh, wow, ouch, etc), Positive and negative words, positive intensifiers (e.g. lovely, surely, etc.), negative intensifiers (e.g. awfully, bitterly etc.), bigrams, trigrams and unigrams, we decided to take the sentiment score using the SentimentIntensityAnalyzer of such words and rightfully append the count of such words to their respective lists.
- For finding nouns and verbs, we chose the Parts of Speech tags from nltk.
- We have also considered a feature called “polarity flips” where in even if the word is negative and the polarity remains positive, and vice versa, we call it flip. We have maintained this in the form of flip count.

These 16 features extracted were divided in to 3 categories:
- Text expression-based features – This category includes counts of features like Nouns, Verbs, Exclamations, Question marks, Ellipses and Capital words.
- Emotion-based features – This category includes counts of features like Positive Intensifiers, Negative Intensifiers, Repeat Letters, Bigrams and trigrams.
- Contrast-based features – This category includes counts of features like Polarity flips, Positive words and Negative words.

Each of these categories were made to undergo 5-fold cross validation each, for each of the 3 classifiers given above. Metrics such as Accuracy, Precision, Recall and F1 score were used.

## Topic Modeling
We also performed topic modeling on the reddit data. To identify the topic for the sarcastic comments, we only selected the rows with sarcastic comments. We combined the comment and the parent comment fields to form a single field. Parent comment was included to get the idea of the topics to which the sarcastic comment was passed.  
We passed these combined comments to the topic modeling engine in the pyspark with parameters as k=3 (number of topics) and max iterations of 50.

Below are the results for obtained after topic modeling:

|   Topic  | Words                                                    |
|:--------:|----------------------------------------------------------|
| Politics | trump, government, hillary, obama, country, world, think |
| Racism   | white, people, racist, black, good, right, police        |
| Sports   | game, playing, people, players, better, time, pretty     |

## Conclusion
Sarcastic and non-sarcastic comments thus, have different linguistic characteristics. Through this project ,we have attempted to show that just increase in the number of features alone does not necessarily pave the way to achieve high accuracy, but selection of the right set of features proves to be the basis of detecting sarcasm. These set of features will change with the change in dataset and data source.
