import os
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

train = pd.read_csv(
    "/Users/rishimalve/Documents/Masters/Sem-3/CS 657/train-balanced-sarcasm.csv")

train.dropna(subset=['comment'], inplace=True)

################################ label count ################################

label_count = train['label'].value_counts()
trace = go.Bar(x=label_count.index, y=label_count.values, marker=dict(
    color=label_count.values, colorscale='RdBu', reversescale=True), )

layout = go.Layout(
    title='Label Count',
    font=dict(size=18)
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="TargetCount")


################################ label distribution ################################

labels = (np.array(label_count.index))
sizes = (np.array((label_count / label_count.sum()) * 100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title='Label Distribution',
                   font=dict(size=18), width=800, height=600, )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="usertype")


################################ Rate of Sarcasm ################################

df = train
df['date'] = pd.to_datetime(df['date'], yearfirst=True)
df['year'] = df['date'].apply(lambda d: d.year)
df.drop(['date', 'created_utc'], axis=1, inplace=True)

comments_by_year = df.groupby('year')['label'].agg([np.sum, np.mean])

plt.figure(figsize=(8, 6))
comments_by_year['mean'].plot(kind='line')
plt.ylabel('Mean Sarcasm')
plt.title('Rate of Sarcasm on Reddit')
plt.show()


################################ Distribution of Scores for Sarcastic and Non-Sarcastic Comments ################################

mean = df['score'].mean()
std = df['score'].std()

plt.figure(figsize=(8, 6))
df[(df['score'].abs() < (10 - ((df['score'].abs() - mean) / std))) &
   (df['label'] == 1)]['score'].hist(alpha=0.5, label='Sarcastic')
df[(df['score'].abs() < (10 - ((df['score'].abs() - mean) / std))) &
   (df['label'] == 0)]['score'].hist(alpha=0.5, label='Not Sarcastic')
plt.yscale('linear')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.legend()
plt.title('Scores for Sarcastic vs. None-Sarcastic Comments')
plt.show()


################################ rate of sarcastic and non sarcastic comments per subreddit ################################

sns.set()
subreddits_to_plot = train.subreddit.value_counts().head(30).index

plot = sns.countplot(x='subreddit', data=train[train.subreddit.isin(
    subreddits_to_plot)], hue='label')
_ = plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
plt.show()


################################ frequent words ################################

train1_df = train[train["label"] == 1]
train0_df = train[train["label"] == 0]

## custom function for ngram generation ##


def generate_ngrams(text, n_gram=2):
    token = [token for token in text.lower().split(
        " ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##


def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation='h',
        marker=dict(
            color=color,
        ),
    )
    return trace


## Get the bar chart from neutral text ##
freq_dict = defaultdict(int)
for sent in train0_df["comment"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

## Get the bar chart from sarcasm text ##
freq_dict = defaultdict(int)
for sent in train1_df["comment"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

# Creating two subplots
fig = tls.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                        subplot_titles=["Frequent words of neutral text",
                                        "Frequent words of sarcasm text"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900,
                     paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')


################################ word cloud for sarcastic words ################################

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0, 16.0), title=None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black', stopwords=stopwords, max_words=max_words,
                          max_font_size=max_font_size, random_state=42, width=800, height=400, mask=mask)
    wordcloud.generate(str(text))

    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors),
                   interpolation="bilinear")
        plt.title(title, fontdict={'size': title_size,
                                   'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud)
        plt.title(title, fontdict={
                  'size': title_size, 'color': 'black', 'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()
    plt.show()


plot_wordcloud(train1_df["comment"], title="Word Cloud of Comment")


sarcastic_comments = str(train[train['label'] == 1]['comment'])
plt.figure(figsize=(12, 12))
word_cloud = WordCloud(stopwords=STOPWORDS)
word_cloud.generate(sarcastic_comments)
plt.imshow(word_cloud)


################################ most sarcastic users ################################

train.groupby('author')['label'].agg([np.sum,np.mean,np.size]).sort_values(by='sum',ascending=False).head(5)