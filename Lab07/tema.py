import numpy as np
import random
import pymc as pm
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv('./auto-mpg.csv')
data1 = data['horsepower'] == '?'
data = data[~data1]
x = data['horsepower'].to_numpy()
x = x.astype(int)
xnorm = x
x = (x-x.mean())/(x.std())
y = data['mpg'].to_numpy()
ynorm = y
y = (y-y.mean())/(y.std())
plt.scatter(x, y)
plt.xlabel('horsepower')
plt.ylabel('mpg', rotation=0)
plt.show()

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1)
    epsilon = pm.HalfCauchy('epsilon', 5)
    u = pm.Deterministic('u', alpha + beta * xnorm)
    y_pred = pm.Normal('y_pred', mu=u, sigma=epsilon, observed=ynorm)

    idata = pm.sample(100, tune=100, chains=4, return_inferencedata=True, cores=1)
    posterior_g = idata.posterior.stack(samples={'chain', 'draw'})

    alpha_m = posterior_g['alpha'].mean().item()
    beta_m = posterior_g['beta'].mean().item()
    draws = range(0, posterior_g.samples.size, 10)
    _, ax= plt.subplots()
    ax.scatter(xnorm, ynorm)
    plt.plot(xnorm, posterior_g['alpha'][draws].values+posterior_g['beta'][draws].values*xnorm[:,None], c='gray', alpha=0.5, ax=ax)
    plt.plot(xnorm, alpha_m+beta_m*x, c='k', label=f'ynorm={alpha_m:.2f}+{beta_m:.2f}*xnorm', ax=ax)
    plt.xlabel('horsepower')
    plt.ylabel('mpg', rotation=0)
    plt.legend()
    plt.show()

