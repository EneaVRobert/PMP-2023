import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats


def func(obv, theta):
    with pm.Model() as model:
        n = pm.Poisson("n", 10)
        y1 = pm.Binomial("y1", n=n, p=theta, observed=obv)
        idata_1 = pm.sample(1000, return_inferencedata=True, cores=1)
        ax1 = pm.sample_posterior_predictive(idata_1, model=model, extend_inferencedata=True)
        az.plot_posterior(ax1)


func(0, 0.2)
func(0, 0.5)
func(5, 0.2)
func(5, 0.5)
func(10, 0.2)
func(10, 0.5)
plt.show()
