import numpy as np
import random
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

# ex1

a1 = stats.expon(0, 1 / 4)
a2 = stats.expon(0, 1 / 6)

a3 = [a1, a2]
draw1 = np.random.choice([0, 1], 10000, p=[0.4, 0.6])

y = [a3[i].rvs(1) for i in draw1]

#az.plot_posterior({'y': y})
#plt.show()

# ex2

b1 = stats.gamma(4, 0, 1 / 3)
b2 = stats.gamma(4, 0, 1 / 2)
b3 = stats.gamma(5, 0, 1 / 2)
b4 = stats.gamma(5, 0, 1 / 3)

b5 = [b1, b2, b3, b4]

draw2 = np.random.choice([0,1,2,3], 10000, p=[0.25,0.25,0.3,0.2])

z = [b5[i].rvs() for i in draw2] + stats.expon.rvs(0,1/4,10000)

num = 0
for i in z:
    if i > 3:
        num = num+1

#print(100*float(num)/float(10000))
#az.plot_posterior({'z': z})
#plt.show()

#ex3

for mix in ["ss", "sb", "bs", "bb"]:
    count1 = np.zeros(100)
    for i in range(100):
        count = []
        for j in range(10):
            drawc1 = "s" if random.random() > 0.5 else "b"
            drawc2 = "s" if random.random() > 0.3 else "b"
            drawc3 = drawc1+drawc2
            count.append(drawc3)
        for j in range(10):
            print(count[j])
            if count[j] == mix:
                count1[i] = count1[i]+1
    az.plot_posterior({'count': count1})
    plt.show()