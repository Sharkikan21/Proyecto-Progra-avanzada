import dash
from dash import dcc, html
import plotly.express as px
import arviz as az
import pandas as pd
import numpy as np
import blackhole

# Data
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

def create_dash_app(trace):
    app = dash.Dash(__name__)

    df_trace = az.extract(trace).to_dataframe()
    fig = px.scatter(df_trace, x="alpha", y="beta", title="Posterior samples")

    app.layout = html.Div([
        html.H1("Visualización de PyMC"),
        dcc.Graph(figure=fig)
    ])

    return app

if __name__ == '__main__':
    with blackhole.model(obsx=data['obsx'], errx=data['errx'], obsy=data['obsy'], erry=data['erry']) as model:
     trace = blackhole.pm.sample(
        draws=10000,         # total samples después del warmup (equivale a iter - warmup)
        tune=5000,           # warmup
        chains=3,
        cores=3,
        thin=10,
        return_inferencedata=True
    )
    app = create_dash_app(trace)
    app.run_server(debug=True)