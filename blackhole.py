import pymc as pm

# modelo para encontrar los valores de la regresion
def model(obsx, errx, obsy, erry):
    N = len(obsx)
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=1000)
        beta = pm.Normal("beta", mu=0, sigma=1000)
        epsilon = pm.Gamma("epsilon", alpha=0.001, beta=0.001)
        
        # Latent true values
        x = pm.Normal("x", mu=0, sigma=1000, shape=N)
        y = pm.Normal("y", mu=0, sigma=1000, shape=N)
        
        # Observed data likelihoods
        obsx_likelihood = pm.Normal("obsx", mu=x, sigma=errx, observed=obsx)
        
        # Regression model for y
        mu_y = alpha + beta * x
        y_likelihood = pm.Normal("y_likelihood", mu=mu_y, sigma=epsilon, shape=N)
        
        obsy_likelihood = pm.Normal("obsy", mu=y, sigma=erry, observed=obsy)
        
    return model

