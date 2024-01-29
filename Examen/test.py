import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats

#Subiectul 1

#incarcare date in pandas dataframe
data = pd.read_csv('./Titanic.csv')
#eliminarea randurilor care contin date care lipsesc din cele relevante
data = data.dropna(axis=0)

#extragerea datelor relevante(Pclass, Age, Survived ca label)
x_n = ['Pclass', 'Age']
x_1 = data[x_n].values
y_1 = data['Survived']

#implementare model regresie logistica multipla
with pm.Model() as model_1:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=2, shape=len(x_n))

    u = alpha + pm.math.dot(x_1, beta)
    o = pm.Deterministic('o', 1 / (1 + pm.math.exp(-u)))
    bd = pm.Deterministic('bd', -alpha / beta[1] - beta[0] / beta[1] * x_1[:, 0])

    y1 = pm.Bernoulli('y1', p=o, observed=y_1)

    idata_1 = pm.sample(2000, return_inferencedata=True, cores=1)

    #vizualizare date si frontiera de decizie
    idx = np.argsort(x_1[:, 0])
    bd1 = idata_1.posterior['bd'].mean(("chain", "draw"))[idx]
    plt.scatter(x_1[:, 0], x_1[:, 1], c=[f'C{x}' for x in y_1])
    plt.plot(x_1[:, 0][idx], bd1, color='k')
    az.plot_hdi(x_1[:, 0], idata_1.posterior['bd'], color='k')
    plt.xlabel(x_n[0])
    plt.ylabel(x_n[1])

#Subpunctul c
#Consider ca variabila ce influenteaza cel mai mult rezultatul este

#Subiectul 2

#N = nr de iteratii, K = nr de estimari pt media folosita
N = 10000
k = 30
# distributie geometrica cu parametrii 0.3, 0.5
x_k = stats.geom.rvs(0.3, size=k)
y_k = stats.geom.rvs(0.5, size=k)
x = stats.geom.rvs(0.3, size=N)
y = stats.geom.rvs(0.5, size=N)
#aplicam conditia la inside
inside = x > y**2
inside_k = x_k > y_k**2
#media peste 30 de valori(k=30)
val = inside_k.mean()
#estimarea metodei Monte Carlo
estimate = inside.sum()*4/N
#eroarea
error = abs((estimate - val)/val) * 100
#plot
outside = np.invert(inside)
plt.figure(figsize=(8,8))
plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
plt.plot(0, 0, label=f'estimate*= {estimate:4.3f}\n error = {error:4.3f}', alpha=0)
plt.axis('square')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)
plt.show()

