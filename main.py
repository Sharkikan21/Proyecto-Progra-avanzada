import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import MCMC, NUTS
import blackhole

# Cargar datos
path_to_data = 'https://raw.githubusercontent.com/astrobayes/BMAD/refs/heads/master/data/Section_10p1/M_sigma.csv'
df = pd.read_csv(path_to_data)
print(df.head())

# Formatear datos
data = {
    'N': len(df),
    'obsx': jnp.array(df['obsx'].values),
    'errx': jnp.array(df['errx'].values),
    'obsy': jnp.array(df['obsy'].values),
    'erry': jnp.array(df['erry'].values)
}

# Iniciar y ejecutar MCMC
rng_key = random.PRNGKey(0)
kernel = NUTS(blackhole.model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
mcmc.run(rng_key, **data)

# Resultados
mcmc.print_summary()
samples = mcmc.get_samples()

# Crea un DataFrame para cada parámetro para facilitar el plot
df_alpha = pd.DataFrame({'alpha': samples['alpha']})
df_beta = pd.DataFrame({'beta': samples['beta']})
df_epsilon = pd.DataFrame({'epsilon': samples['epsilon']})

# Inicializa la app Dash
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Distribuciones posteriores de parámetros (NumPyro + Dash)'),

    dcc.Graph(
        id='hist-alpha',
        figure=px.histogram(df_alpha, x='alpha', nbins=50, title='Posterior de alpha')
    ),

    dcc.Graph(
        id='hist-beta',
        figure=px.histogram(df_beta, x='beta', nbins=50, title='Posterior de beta')
    ),

    dcc.Graph(
        id='hist-epsilon',
        figure=px.histogram(df_epsilon, x='epsilon', nbins=50, title='Posterior de epsilon')
    ),
])

if __name__ == '__main__':
    app.run(debug=True)