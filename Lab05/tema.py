import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

count_data = np.loadtxt("trafic.csv", delimiter=',', dtype=int)
n_count_data = len(count_data)
with pm.Model() as model:
    alpha = 1.0/count_data[:, 1].mean()

    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)
    lambda_4 = pm.Exponential("lambda_4", alpha)
    lambda_5 = pm.Exponential("lambda_5", alpha)

    interval1 = pm.Normal("interval1", 60*3)
    interval2 = pm.Normal("interval2", 60*12)
    interval3 = pm.Normal("interval3", 60*15)
    interval4 = pm.Normal("interval4", 60*20)

    tau1 = pm.DiscreteUniform("tau1", lower=1, upper=interval1)
    tau2 = pm.DiscreteUniform("tau2", lower=tau1, upper=interval2)
    tau3 = pm.DiscreteUniform("tau3", lower=tau2, upper=interval3)
    tau4 = pm.DiscreteUniform("tau4", lower=tau3, upper=interval4)

    with model:
        idx = np.arange(n_count_data)
        lmbd1 = pm.math.switch(tau1 > idx, lambda_1, lambda_2)
        lmbd2 = pm.math.switch(tau2 > idx, lmbd1, lambda_3)
        lmbd3 = pm.math.switch(tau3 > idx, lmbd2, lambda_4)
        lmbd4 = pm.math.switch(tau4 > idx, lmbd3, lambda_5)
        observation = pm.Poisson("obs", lmbd4, observed=count_data[:, 1])
        trace = pm.sample(10, cores=1)
        az.plot_posterior(trace)
        plt.show()

