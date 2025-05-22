import numpyro
from numpyro import distributions as dist


def model(N, obsx, errx, obsy, erry):
    # Priors
    alpha = numpyro.sample("alpha", dist.Normal(0, 1000))
    beta = numpyro.sample("beta", dist.Normal(0, 1000))
    epsilon = numpyro.sample("epsilon", dist.Gamma(0.001, 0.001))

    # True latent variables
    x = numpyro.sample("x", dist.Normal(0, 1000).expand([N]))
    y = numpyro.sample("y", dist.Normal(0, 1000).expand([N]))

    # Likelihoods
    numpyro.sample("obsx", dist.Normal(x, errx), obs=obsx)
    numpyro.sample("y_model", dist.Normal(alpha + beta * x, epsilon), obs=y)
    numpyro.sample("obsy", dist.Normal(y, erry), obs=obsy)