import numpy as np
import random
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

client_nr = stats.poisson.rvs(20, size=1)

command_nr = stats.norm.rvs(loc=2, scale=0.5, size=20)

def calc(guess):
    rand_gen = stats.expon.rvs(loc=guess, size=20)
    count = 0
    for element in rand_gen:
        if element < 15:
            count = count + 1
    if count / 20 * 100 > 95:
        return calc(guess + 1)
    else:
        return guess


alpha = calc(1)
print(alpha)
cook_nr = stats.expon.rvs(loc=alpha, size=20)

az.plot_posterior({'serve time': cook_nr})
plt.show()
