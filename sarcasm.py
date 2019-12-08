import os
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go

train = pd.read_csv(
    "/Users/rishimalve/Documents/Masters/Sem-3/CS 657/final_project/data/train-balanced-sarcasm.csv")

## label count ##
cnt_srs = train['label'].value_counts()
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale='Picnic',
        reversescale=True
    ),
)

layout = go.Layout(
    title='Label Count',
    font=dict(size=18)
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="TargetCount")

## label distribution ##
labels = (np.array(cnt_srs.index))
sizes = (np.array((cnt_srs / cnt_srs.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Label Distribution',
    font=dict(size=18),
    width=800,
    height=600,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="usertype")
