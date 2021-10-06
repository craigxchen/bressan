import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
# import pandas as pd

def f(x,y):
    return (x+y)%1, y

def g(x,y):
    return x, (x+y)%1

def cat_map(x,y):
    return g(*f(x,y))

def F(x,y,k: int):
    return (1-k*np.abs(y-0.5))%1, 0

def G(x,y,k: int):
    return 0, (1-k*np.abs(x-0.5))%1

def flow(x,y,t,k=2):
    if np.floor(t)%2 == 0:
        return F(x,y,k)
    else:
        return G(x,y,k)

N = 1
T = 1000
timesteps = np.linspace(0,T,T*N)

x0, y0 = (1/np.sqrt(2), 1/np.sqrt(2))

def simulate(x, y, function, timesteps):
    x_hist = [x]
    y_hist = [y]
    for i in range(len(timesteps)):
        dx, dy = function(x,y,i)
        x,y = (x + dx/N)%1, (y + dy/N)%1
        x_hist.append(x)
        y_hist.append(y)
    return x_hist, y_hist

xx_hist, yy_hist = np.array(simulate(x0, y0, lambda x,y,t: flow(x,y,t,k=2), timesteps))

# Create figure
fig = go.Figure(data=[
    go.Scatter(
            x=xx_hist,
            y=yy_hist,
            mode='lines',
            line=dict(width=2, color="blue")
            ),
     go.Scatter(
            x=xx_hist,
            y=yy_hist,
            mode='lines',
            line=dict(width=2, color="blue")
            )
    ],
    frames=[go.Frame(
        data=[go.Scatter(
            x=[xx_hist[k]],
            y=[yy_hist[k]],
            mode="markers",
            marker=dict(color="red", size=10))])

        for k in range(len(xx_hist))
    ],
    layout=go.Layout(
        xaxis=dict(range=[0, 1], autorange=False),
        yaxis=dict(range=[0, 1], autorange=False),
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, {"frame": {"duration": 1, "redraw": False},
                                              "fromcurrent": True,
                                              "transition": {"duration": 0}}]),
                                   dict(label="Pause",
                                        method="animate",
                                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}])])
                     ])
    )

fig.update_geos(resolution=110)

# Create and add slider
# steps = []
# for i in range(len(fig.frames)):
#     step = dict(
#         method="restyle",
#         args=[{"visible": [False] * len(fig.frames)}],  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True
#     steps.append(step)

# sliders = [dict(
#     active=0,
#     steps=steps
# )]

# fig.update_layout(
#     sliders=sliders,
#     xaxis=dict(range=[0, 1], autorange=False),
#     yaxis=dict(range=[0, 1], autorange=False)
# )

plot(fig)