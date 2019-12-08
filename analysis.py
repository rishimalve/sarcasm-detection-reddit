import os
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv(
    "/Users/rishimalve/Documents/Masters/Sem-3/CS 657/train-balanced-sarcasm.csv")

train.dropna(subset=['comment'], inplace=True)

################################ label count ################################

label_count = train['label'].value_counts()
trace = go.Bar(x=label_count.index, y=label_count.values, marker=dict(
    color=label_count.values, colorscale='RdBu', reversescale=True),)

layout = go.Layout(
    title='Label Count',
    font=dict(size=18)
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="TargetCount")


################################ label distribution ################################

labels = (np.array(label_count.index))
sizes = (np.array((label_count / label_count.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title='Label Distribution',
                   font=dict(size=18), width=800, height=600,)
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
df[(df['score'].abs() < (10-((df['score'].abs()-mean)/std))) &
    (df['label'] == 1)]['score'].hist(alpha=0.5, label='Sarcastic')
df[(df['score'].abs() < (10-((df['score'].abs()-mean)/std))) &
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

plot = sns.countplot(x='subreddit', data=train[train.subreddit.isin(subreddits_to_plot)], hue='label')
_ = plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
plt.show()