import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, 3]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster),
                       np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix))
plt.show()

components1 = 2
with pm.Model() as model1:
    p1 = pm.Dirichlet('p1', a=np.ones(components1))
    means1 = pm.Normal('means1', mu=np.array(mix).mean(), sigma=10, shape=components1)
    sd1 = pm.HalfNormal('sigma1', sigma=10)
    y1 = pm.NormalMixture('y1', w=p1, mu=means1, sigma=sd1, observed=np.array(mix))
    idata_mg1 = pm.sample(random_seed=123, return_inferencedata=True, cores=1, idata_kwargs={'log_likelihood': True})

components2 = 3
with pm.Model() as model2:
    p2 = pm.Dirichlet('p2', a=np.ones(components2))
    means2 = pm.Normal('means2', mu=np.array(mix).mean(), sigma=10, shape=components2)
    sd2 = pm.HalfNormal('sigma2', sigma=10)
    y2 = pm.NormalMixture('y2', w=p2, mu=means2, sigma=sd2, observed=np.array(mix))
    idata_mg2 = pm.sample(random_seed=123, return_inferencedata=True, cores=1, idata_kwargs={'log_likelihood': True})

components3 = 4
with pm.Model() as model3:
    p3 = pm.Dirichlet('p3', a=np.ones(components3))
    means3 = pm.Normal('means3', mu=np.array(mix).mean(), sigma=10, shape=components3)
    sd3 = pm.HalfNormal('sigma3', sigma=10)
    y3 = pm.NormalMixture('y3', w=p3, mu=means3, sigma=sd3, observed=np.array(mix))
    idata_mg3 = pm.sample(random_seed=123, return_inferencedata=True, cores=1, idata_kwargs={'log_likelihood': True})

cmp_loocv = az.compare({'model1': idata_mg1, 'model2': idata_mg2, 'model3': idata_mg3},
                       method='stacking', ic='loo', scale='deviance')

print(cmp_loocv)

cmp_waic = az.compare({'model1': idata_mg1, 'model2': idata_mg2, 'model3': idata_mg3},
                      method='stacking', ic='waic', scale='deviance')

print(cmp_waic)
