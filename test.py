import dash
from dash import dcc, html
import plotly.express as px
import arviz as az
import pandas as pd
import numpy as np
import blackhole

path_to_data = 'https://raw.githubusercontent.com/astrobayes/BMAD/refs/heads/master/data/Section_10p1/M_sigma.csv'
df = pd.read_csv(path_to_data)
print(df.head())
data = {
    'N': len(df),
    'obsx': df['obsx'].values,
    'errx': df['errx'].values,
    'obsy': df['obsy'].values,
    'erry': df['erry'].values
}
