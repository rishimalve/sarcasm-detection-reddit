import os
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.pyplot as plt

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

comments_by_year = df.groupby('year')['label'].agg([np.sum,np.mean])

plt.figure(figsize=(8,6))
comments_by_year['mean'].plot(kind='line')
plt.ylabel('Mean Sarcasm')
plt.title('Rate of Sarcasm on Reddit')
plt.show()