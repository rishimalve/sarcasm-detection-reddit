# -*- coding: utf-8 -*-
import os
import pandas as pd
from string import punctuation
import re as re

data = pd.read_csv("data/train-balanced-sarcasm.csv")

data = data[data['label'] == 1]
data = data[data['parent_comment'].notnull()]
data = data[data['comment'].notnull()]
# data["combined"] = data[data['comment']+data['parent_comment']]
data["combined"] = data['comment'] + " " + data['parent_comment']

comments = list(data['combined'])

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)

stopwords = ["dont", "youre", "like", "cant", "said", "yeah", "doesnt", "thats", "all", "want", "six", "just", "less", "being", "indeed", "over", "move", "anyway", "fifty", "four", "not", "own", "through", "using", "go", "only", "its", "‘ll", "before", "one", "whose", "how", "somewhere", "with", "show", "had", "enough", "‘d", "should", "to", "must", "whom", "seeming", "yourselves", "under", "ours", "has", "might", "into", "latterly", "do", "them", "his", "around", "than", "get", "very", "none", "n’t", "cannot", "every", "whether", "they", "front", "during", "thus", "now", "‘ve", "him", "nor", "name", "regarding", "several", "hereafter", "did", "always", "who", "whither", "'ve", "this", "someone", "either", "each", "become", "thereupon", "sometime", "side", "two", "’re", "therein", "twelve", "because", "often", "ten", "our", "doing", "some", "back", "used", "up", "namely", "towards", "are", "further", "beyond", "ourselves", "out", "even", "will", "what", "still", "for", "bottom", "mine", "‘m", "since", "please", "forty", "per", "yet", "everything", "behind", "does", "various", "above", "between", "it", "neither", "seemed", "ever", "across", "she", "somehow", "be", "we", "full", "never", "sixty", "however", "here", "quite", "were", "whereupon", "nowhere", "although", "others", "alone", "re", "along", "fifteen", "by", "both", "about", "last", "would", "anything", "’ve", "via", "many", "could", "thence", "put", "against", "keep", "where", "amount", "became", "'re", "hence",
             "onto", "or", "first", "among", "already", "afterwards", "didnt", "going", "maybe", "isnt", "formerly", "seems", "ca", "within", "while", "whatever", "except", "down", "hers", "everyone", "done", "least", "another", "’d", "whoever", "moreover", "throughout", "anyhow", "yourself", "three", "from", "her", "eleven", "twenty", "top", "there", "due", "been", "next", "anyone", "few", "much", "call", "therefore", "then", "thru", "themselves", "hundred", "was", "until", "empty", "more", "himself", "elsewhere", "mostly", "on", "‘re", "am", "becoming", "hereby", "'s", "amongst", "else", "part", "everywhere", "too", "‘s", "herself", "former", "those", "he", "me", "myself", "made", "’m", "these", "say", "us", "besides", "nevertheless", "below", "anywhere", "nine", "can", "of", "your", "’s", "toward", "my", "something", "and", "n‘t", "whereafter", "whenever", "give", "almost", "wherever", "is", "beforehand", "herein", "an", "n't", "as", "itself", "at", "have", "in", "seem", "any", "if", "again", "’ll", "'ll", "thereby", "no", "perhaps", "latter", "meanwhile", "when", "whence", "same", "wherein", "beside", "also", "that", "other", "take", "which", "becomes", "you", "really", "nobody", "unless", "whereas", "see", "though", "may", "thereafter", "after", "upon", "most", "hereupon", "'d", "eight", "but", "serious", "nothing", "such", "'m", "why", "a", "off", "whereby", "third", "i", "whole", "noone", "sometimes", "well", "together", "yours", "their", "rather", "without", "so", "five", "the", "otherwise", "make", "once", "'nt"]

cleaned_comments = []
for comment in comments:
    comment = comment.lower()
    comment = re.sub(emoji_pattern, "", comment)
    comment = re.sub(r"http\S+", "", comment)
    comment = comment.strip()
    words = comment.split()
    cleaned_comment = ''
    for word in words:
        word = ''.join(c for c in word if c not in punctuation)
        if word not in stopwords:
            cleaned_comment += word + " "
    cleaned_comment = cleaned_comment.strip()
    cleaned_comments.append(cleaned_comment)

data['comment'] = cleaned_comments
data = data[data['comment'] != '']
data = data.drop_duplicates(subset='comment')
data = data.reset_index()
data.drop(["index", "label", "author", "subreddit", "score",
           "ups", "downs", "date", "created_utc", "parent_comment"], axis=1, inplace=True)
data.to_csv(r'data/input_TM.csv')
