import numpy as np
from scipy import stats
import pymc as pm
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(1)

#ex1 pct 1
p1won = 0
p2won = 0

for j in range(10000):
    starting = np.random.choice([0, 1], 1, p=[0.5, 0.5])
    rez1 = 0
    rez2 = 0

    a1 = stats.uniform(0, 1)
    a2 = stats.uniform(0, 1)
    starting1 = [a1.rvs(1) for i in starting]
    # determinam care jucator incepe primul
    if starting1[0][0] > 0.5:
        draw1 = np.random.choice([0, 1], 1, p=[0.5, 0.5])

        y1 = [a1.rvs(1) for i in draw1]
        if y1[0][0] > 0.5:
            rez1 = 1
        else:
            rez1 = 0

        draw2 = np.random.choice([0, 1], rez1 + 1, p=[0.5, 0.5])

        y2 = [a2.rvs(1) for i in draw2]

        for i in range(len(y2)):
            if y2[i][0] > 1 / 3:
                rez2 += 1

        if rez1 > rez2:
            p1won += 1
        else:
            p2won += 1
    else:
        draw1 = np.random.choice([0, 1], 1, p=[0.5, 0.5])

        y1 = [a1.rvs(1) for i in draw1]
        if y1[0][0] > 1 / 3:
            rez2 = 1
        else:
            rez2 = 0

        draw2 = np.random.choice([0, 1], rez2 + 1, p=[0.5, 0.5])

        y2 = [a2.rvs(1) for i in draw2]

        for i in range(len(y2)):
            if y2[i][0] > 0.5:
                rez1 += 1

        if rez1 > rez2:
            p1won += 1
        else:
            p2won += 1

print(p1won)
print(p2won)
# al doilea jucator are sanse mai mari de castig

# ex1 pct2
model = BayesianNetwork([('I', 'C'), ('I', 'P'), ('P', 'C')])

cpd_ci = TabularCPD(variable='I', variable_card=2, values=[[0.5], [0.5]])
cpd_p = TabularCPD(variable='P', variable_card=2, values=[[0.5, 1 / 3], [0.5, 2 / 3]], evidence=['I'], evidence_card=[2])

cpd_coins = TabularCPD(variable='C', variable_card=3, values=[
    [1 / 3, 1 / 3 * 1 / 3,  1 / 2, 1 / 2 * 1 / 2],
    [2 / 3, 1 / 3 * 2 / 3,  1 / 2, 1 / 2 * 1 / 2],
    [0,  2 / 3 * 2 / 3, 0, 1 / 2 * 1 / 2]], evidence=['I', 'P'], evidence_card=[2, 2])

model.add_cpds(cpd_ci, cpd_p, cpd_coins)

#ex1 pct 3
#infer = VariableElimination(model)
#result = infer.query(variables=['I'], evidence={'C': 1})
#print(result)

#ex2 pct 1
trials = 100
u = 100
sigma = 0.2
data = stats.norm.rvs(loc=u, scale=1/(sigma**2), size=trials)

#ex2 pct 2
with pm.Model() as model:

    sigma = pm.HalfCauchy('sigma', 5)
    u = pm.Normal('u', 10)
    model = pm.Normal("inference", mu=u, sigma=sigma)

    #Alegerea a fost facuta deoarece datele se afla sub forma unei distributii normale,
    # modelul urmand sa fie aceeasi distributie, pe cand



