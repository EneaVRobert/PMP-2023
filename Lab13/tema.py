import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd

dat1 = az.load_arviz_data("centered_eight")
dat2 = az.load_arviz_data("non_centered_eight")

summaries = pd.concat([az.summary(dat1, var_names=['mu', 'tau']), az.summary(dat2, var_names=['mu', 'tau'])])
print(summaries)
az.plot_autocorr(dat1, var_names=['mu', 'tau'])
az.plot_autocorr(dat2, var_names=['mu', 'tau'])
plt.show()
plt.close()