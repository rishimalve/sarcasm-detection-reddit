#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:56:00 2019

@author: pranjalihugay
"""

import pandas as pd
import csv
import os
import re
import nltk
import string
import numpy as np
import constants
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ekphrasis.classes.segmenter import Segmenter
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer



#def main():
filepath = "./train-balanced-sarcasm.csv"
reddit_data = read_data(filepath)
reddit_data.dropna(subset=['comment'], inplace=True)#from 1010826 to 1010773
labels = list(reddit_data['label'].values)
comments = list(reddit_data['comment'].values)
sia = SentimentIntensityAnalyzer()
ps = PorterStemmer()
wnl = WordNetLemmatizer()
stoplist = list(set(stopwords.words("english")))
exclamation_count = []
questionmark_count = []
ellipsis_count = []
interjection_count = []
capitalWords_count = []
repeatLetter_counts = []
positive_wordcount = []
negative_wordcount = []
polarityFlip_count = []
noun_count = []
verb_count = []
positive_intensifiercount = []
negative_intensifiercount = []
bigrams_positive_sentiment = []
bigrams_negative_sentiment = []
trigrams_positive_sentiment = []
trigrams_negative_sentiment = []
#passive_aggressive_count = []
interjections = ["aah","ack","agreed","ah","aha","ahem","alas","all right","amen","argh","as if","aw",
                 "ay","aye","bah","blast","boo hoo","bother","boy","brr","by golly","bye","cheerio",
                 "cheers","chin up","come on","crikey","curses","dear me","doggone","drat","duh",
                 "easy does it","eek","egads","er","exactly","fair enough","fiddle-dee-dee","fiddlesticks",
                 "fie","foo","fooey","g'day","gadzooks","gah","gangway","gee","gee whiz","geez",
                 "gesundheit","get lost","get outta here","go on","good","good golly","good job",
                 "gosh","gracious","great","grr","gulp","ha","ha-ha","hah","hallelujah","harrumph",
                 "haw","hee","here","hey","hmm","ho hum","hoo","hooray","hot dog","how","huh","hum",
                 "humbug","hurray","huzza","I say","ick","is it","ixnay","jeez","just a sec",
                 "just kidding","just wondering","kapish","la","la-di-dah","lo","long time","look",
                 "look here","lordy","man","meh","mmm","most certainly","mymy","my word","nah","naw",
                 "never","no","no can do","no thanks","no way","nooo","not","nuts","oh","oh no","oh-oh",
                 "oho","okay","okey-dokey","om","oof","ooh","oopsey","oops","over","oy","oyez","peace",
                 "pew","pff","phew","pish posh","psst","ptui","quite","rah","rats","ready","right",
                 "right on","roger","roger that","rumble","say","see ya","shame","shh","shoo","shucks",
                 "sigh","sleep tight","snap","sorry","sssh","sup","ta","ta ta","ta-da","take that",
                 "tally ho","tch","thanks","there","there there","time out","toodles","touche","tsk",
                 "tsk-tsk","tut","tut-tut","ugh","uh","uh-oh","um","ur","urgh","very nice","very well",
                 "voila","vroom","wah","well","well done","well,  well","what","whatever","whee","when","whew",
                 "whoa","whoo","whoopee","whoops","whoopsy","why","word","wow","wuzzup","ya","yea",
                 "yeah","yech","yikes","yippee","yo","yoo-hoo","you bet","you don't say","you know",
                 "yow","yum","yummy","zap","zounds","zowie","zzz"]
intensifiers = ["amazingly","awfully","bitterly","critically","dangerously","deeply","dreadfully	",
                "especially","exceedingly","extremely","greatly","highly","hopelessly","horribly",
                "hugely","incredibly","particularly","really","remarkably","seriously","strikingly",
                "surprisingly","suspiciously","terribly","unbelievably","very","violently",
                "wonderfully","faintly","fairly","mildly","moderately","pretty","quite","rather",
                "reasonably","slightly","somewhat","almost","nearly","exclusively","partly","fully",
                "predominantly","largely","primarily","mainly","roughly","mostly","absolutely",
                "purely","altogether","quite","completely","simply","entirely","totally","perfectly",
                "utterly"]
#most_common_unigrams = find_common_unigrams(reddit_data,wnl,stoplist)
#print(most_common_unigrams)
i=0
comm = []
for c in comments:
    i+=1
    print(i)
    #tokenizing every comment
    tokens = clean_data(c,wnl,stoplist)    
    if len(tokens)!=0: #total records = 1003902
        p = punctuations_counter(c, ['!', '?', '...'])
        exclamation_count.append(p['!'])
        questionmark_count.append(p['?'])
        ellipsis_count.append(p['...'])
        interjection_count.append(interjections_counter(c,interjections))
        capitalWords_count.append(capitalWords_counter(tokens))
        repeatLetter_counts.append(repeatLetterinWords_counter(c))
        x = polarityFlip_counter(tokens)
        positive_wordcount.append(x[0])
        negative_wordcount.append(x[1])
        polarityFlip_count.append(x[-1])
        x = grammar_count(tokens)
        noun_count.append(x[0])
        verb_count.append(x[1])
        x = intensifier_counter(tokens,intensifiers)
        positive_intensifiercount.append(x[0])
        negative_intensifiercount.append(x[1])
        x = skip_grams(tokens, 2, 0)
        bigrams_positive_sentiment.append(x[0])
        bigrams_negative_sentiment.append(x[1])
        y = skip_grams(tokens, 3, 0)
        trigrams_positive_sentiment.append(x[0])
        trigrams_negative_sentiment.append(x[1])
        #passive_aggressive_count.append(passive_aggressive_counter(c))
    else:
        pass


def read_data(filepath):
    reddit_comments = pd.read_csv(filepath)
    reddit_comments = reddit_comments.drop(["author","subreddit","score","ups","downs","date","created_utc","parent_comment"]
    ,axis = 1)
    return reddit_comments

#tokenize,clean each sentences of punctuation,remove stopwords,lemmatize and return clean comments
def clean_data(comment,lemm, stoplist):
    tokens = nltk.word_tokenize(comment) #making tokens from sentence
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word.lower() not in stoplist]
    tokens = [lemm.lemmatize(word) for word in tokens]
    return tokens

#check for occurences of !/?/... in the comment
def punctuations_counter(comment, punct_list):
    punct_count = {}
    for p in punct_list:
        punct_count.update({p: comment.count(p)})
    return punct_count

#check for comment to contain any of the interjections given in list and keep count
def interjections_counter(comment,interjections):
    interjection_count = 0
    for inter in interjections:
        if inter in comment.lower():
            return inter


#find positive and negative intense words
def intensifier_counter(tokens,intensifiers):
    positive = []
    negative = []
    for word in tokens:
        if word in intensifiers:
            ss_in = sia.polarity_scores(word)
            if (ss_in["neg"] == 1.0):
                positive.append(word)
            if (ss_in["pos"] == 1.0):
                negative.append(word)
    return positive, negative

#finding flips in polarity using positive and negative words
def polarityFlip_counter(tokens):
    pos = False
    neg = False
    pos_word = []
    neg_word = [] 
    flips = 0
    for words in tokens:
        ss = sia.polarity_scores(words)
        if ss["neg"] == 1.0:
            neg = True
            if pos:
                flips += 1
                positive = False
            neg_word.append(words)
        elif ss["pos"] == 1.0:
            pos = True
            if neg:
                flips += 1
                neg = False
            pos_word.append(words)        
    return pos_word, neg_word, flips

#check for no of capital words a comment has
def capitalWords_counter(tokens):
    upperCase = 0
    for words in tokens:
        if words.isupper():
            return words  

#check if any word in the comment has repeated letters eg:whaaaaat and keep count of such words
def repeatLetterinWords_counter(tweet):
    repeat_letter_words = 0
    matcher = re.compile(r'(.)\1*')
    repeat_letters = [match.group() for match in matcher.finditer(tweet)]
    for segments in repeat_letters:
        if len(segments) >= 3 and str(segments).isalpha():
            return segments


#find nouns,verbs
def grammar_count(tokens):
    Tagged = nltk.pos_tag(tokens)
    nouns = ['NN', 'NNS', 'NNP', 'NNPS']
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    Nouns = []
    Verbs = []
    no_words = len(tokens)
    for i in range(0, len(Tagged)):
        if Tagged[i][1] in nouns:
            Nouns.append(Tagged[i][0])
        if Tagged[i][1] in verbs:
            Verbs.append(Tagged[i][0])
    return Nouns, Verbs

#skip-grams 
def skip_grams(tokens, n, dist):
    skip_gram_value = 0
    pos_words = []
    neg_words = []
    arr = [x for x in nltk.skipgrams(tokens, n, dist)]
    for j in range(len(arr)):
        for k in range(n):
            ss = sia.polarity_scores(arr[j][k])
            if (ss["pos"] == 1):
                skip_gram_value += 1
                pos_words.append(arr[j][k])
            if (ss["neg"] == 1):
                skip_gram_value -= 1
                neg_words.append(arr[j][k])
    return pos_words, neg_words


# =============================================================================
# #find total passive-aggressive statements in each comment
# def passive_aggressive_counter(comment):
#    pass_aggr_sentence_count = 0
#    new_sentence = []
#    i = 0
#    for j in range(len(comment)):
#        if '.' != comment[j]:
#            i += 1
#            new_sentence.append(comment[j])
#        else:
#            makeitastring = ''.join(map(str, new_sentence))
#            tokens = (nltk.word_tokenize(makeitastring))
#            if len(tokens)<3 and len(tokens) >0:
#                pass_aggr_sentence_count += 1
#            if i == len(comment):
#                return 0
#            new_sentence = []
#    return pass_aggr_sentence_count
# =============================================================================

#tfidf vectorizing features
def find_TFIDF(features):
    vector = TfidfVectorizer(analyzer='word', input='content', stop_words=stopwords.words('english'), ngram_range=(2,2))
    features_tfidf = vector.fit_transform(features)


# Normalize every feature
def normalize(featureList):
    max = np.max(featureList)
    min = np.min(featureList)
    def normalize(x):
        return round(((x-min) / (max-min)),2)
    if max != 0:
        featureList = [x for x in map(normalize, featureList)]
    return featureList


#make a list of features
features = zip(labels,normalize(exclamation_count),normalize(questionmark_count),
normalize(ellipsis_count), normalize(interjection_count),
normalize(capitalWords_count), normalize(repeatLetter_counts),
normalize(positive_wordcount), normalize(negative_wordcount), normalize(polarityFlip_count),
noun_count, verb_count, normalize(positive_intensifiercount),
normalize(negative_intensifiercount), bigrams_sentiment, trigrams_sentiment,
normalize(passive_aggressive_count))

#make a list of features
newfeatures = zip(labels,exclamation_count,questionmark_count,
ellipsis_count, interjection_count,
capitalWords_count, repeatLetter_counts,
positive_wordcount, negative_wordcount, polarityFlip_count,
noun_count, verb_count, positive_intensifiercount,
negative_intensifiercount, bigrams_sentiment, trigrams_sentiment,
passive_aggressive_count)

# Headers for the new feature list   
headers = ["label", "Exclamations", "Question_Marks", "Ellipsis", "Interjections", "Capitals",
           "RepeatLetters", "Positive_word_count", "Negative_word_count", "Polarity_flips",
           "Nouns", "Verbs", "PositiveIntensifiers", "NegativeIntensifiers", "Bigrams", "Trigram",
           "Passive_aggressive_count"]


# Writing headers to the new .csv file
with open("./normfeatures.csv", "w") as header:
    header = csv.writer(header)
    header.writerow(headers)

# Append the feature list to the file
with open("./normfeatures.csv", "a") as feature_csv:
    writer = csv.writer(feature_csv)
    for line in features:
        writer.writerow(line)