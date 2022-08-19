from sklearn.svm import SVC
from individual import Individuo
from utils import gen_population, select_parents_withouth_bias, crossover
from sklearn.model_selection import train_test_split
from support_vector import evaluate_population
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import threading
import gc

def genetic_algorithm_init(
  nPop: int,
  epochs: int,
  X, y) -> Individuo:

  population: list = gen_population(nPop)
  best = Individuo(f=0)
  for i in range(epochs):
    evaluate_population(population, X=X, y=y)

    temp_pop = population.copy()
    temp_pop.sort(key=lambda fit: fit.fitness, reverse=True)

    if temp_pop[0].fitness > best.fitness:
      best = temp_pop[0]

    candidates = select_parents_withouth_bias(population)

    pos = 0
    while pos < len(population):
      proba = random.random()
      if proba < 0.8:
        f = random.choice(candidates)
        m = random.choice(candidates)
        population[pos] = crossover(f, m)
        pos += 1

    for i in population:
      be_mutate: float = random.random()
      if be_mutate < 1/len(population):
        i.mutate()

  return best

# Init 

df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'].replace(['negative', 'positive'], [0, 1], inplace=True)

df_positive = df.loc[df['sentiment'] == 1]
df_negative = df.loc[df['sentiment'] == 0]

frames = [df_positive, df_negative]
df_final = pd.concat(frames)

df_final = df_final.sample(frac=1).reset_index(drop=True)

# Test 

val_test = list(range(1000, 41000, 1000))
tfid = TfidfVectorizer(stop_words='english')

def evaluate_test_size(
  tam: int) -> None:

  ga_df = df_final[:tam]
  X, y = ga_df['review'], ga_df['sentiment']
  best = genetic_algorithm_init(20, 6, X, y)
  print('BEST Fitness: ', best.get_fitness())

  for test_val in val_test:
    val_df = df_final[:test_val]
    _X, _y = val_df['review'], val_df['sentiment']

    # default 
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(_X, _y, test_size=0.3)
    X_train_d = tfid.fit_transform(X_train_d)
    X_test_d = tfid.transform(X_test_d)

    svc_d = SVC()
    svc_d.fit(X_train_d, y_train_d)
    print('DEFAULT: ', svc_d.score(X_test_d, y_test_d))

    # Hiper - GA
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(_X, _y, **best.get_split())
    X_train_g = tfid.fit_transform(X_train_g)
    X_test_g = tfid.transform(X_test_g)

    svc_g = SVC(**best.get_config())
    svc_g.fit(X_train_g, y_train_g)
    print('Hiper-GA: ', svc_g.score(X_test_g, y_test_g))

    gc.collect()


""" 
t2 = threading.Thread(target=evaluate_test_size, args=(300,))
t3 = threading.Thread(target=evaluate_test_size, args=(400,))
t4 = threading.Thread(target=evaluate_test_size, args=(500,))
t5 = threading.Thread(target=evaluate_test_size, args=(600,))
t6 = threading.Thread(target=evaluate_test_size, args=(700,))
t7 = threading.Thread(target=evaluate_test_size, args=(800,))

t2.start()
t2.join()

t3.start()
t3.join()

t4.start()
t4.join()


t5.start()
t5.join()

t6.start()
t6.join()

t7.start()
t7.join()
"""





#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x,y)

plt.show()

plt.savefig('foo.png')






