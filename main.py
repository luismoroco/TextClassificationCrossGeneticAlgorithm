from sklearn.svm import SVC
from individual import Individuo
from utils import gen_population, select_parents_withouth_bias, crossover
from sklearn.model_selection import train_test_split
from support_vector import evaluate_population
from sklearn.feature_extraction.text import TfidfVectorizer
from plot import generate_ga_plot, generate_val_size_plot
import pandas as pd
import random

def genetic_algorithm_init(
  nPop: int,
  epochs: int,
  X, y) -> Individuo:

  population: list = gen_population(nPop)
  best = Individuo(f=0)

  best_each_epochs: list = []

  for i in range(epochs):
    evaluate_population(population, X=X, y=y)

    temp_pop = population.copy()
    temp_pop.sort(key=lambda fit: fit.fitness, reverse=True)

    best_each_epochs.append(temp_pop[0].get_fitness())

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

  return best_each_epochs, best

# Init 

df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'].replace(['negative', 'positive'], [0, 1], inplace=True)

df_positive = df.loc[df['sentiment'] == 1]
df_negative = df.loc[df['sentiment'] == 0]

frames = [df_positive, df_negative]
df_final = pd.concat(frames)

df_final = df_final.sample(frac=1).reset_index(drop=True)

# Test 

POPULATION_SIZE = 20
EPOCHS = 6

val_test = [1000, 2000, 3000, 4000]
index_epochs = list(range(1, EPOCHS+1))

tfid = TfidfVectorizer(stop_words='english')

def evaluate_test_size(
  tam: int) -> None:

  ga_df = df_final[:tam]
  X, y = ga_df['review'], ga_df['sentiment']
  pick_epochs, best = genetic_algorithm_init(POPULATION_SIZE, EPOCHS, X, y)

  print('BEST Fitness: ', best.get_fitness())

  print('GENERANDO PLOT\n')
  generate_ga_plot(
    index=index_epochs,
    picks=pick_epochs,
    size_ga_init=tam
  )

  def_f = []
  hip_f = []

  for test_val in val_test:
    val_df = df_final[:test_val]
    _X, _y = val_df['review'], val_df['sentiment']

    # default 
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(_X, _y, test_size=0.3)
    X_train_d = tfid.fit_transform(X_train_d)
    X_test_d = tfid.transform(X_test_d)

    svc_d = SVC()
    svc_d.fit(X_train_d, y_train_d)

    def_score: float = svc_d.score(X_test_d, y_test_d)
    print('DEFAULT: ', def_score)
    def_f.append(def_score)

    # Hiper - GA
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(_X, _y, **best.get_split())
    X_train_g = tfid.fit_transform(X_train_g)
    X_test_g = tfid.transform(X_test_g)

    svc_g = SVC(**best.get_config())
    svc_g.fit(X_train_g, y_train_g)

    hip_score: float = svc_g.score(X_test_g, y_test_g)
    print('Hiper-GA: ', hip_score)
    hip_f.append(hip_score)

  print('GENERANDO PLOT VAL\n')
  generate_val_size_plot(
    index_tam_val=val_test,
    def_hip=def_f,
    hip_ga=hip_f,
    size_ga_init=tam
  )

init_test_ga = [200, 300, 400, 500, 600, 700, 800, 900]
for tam_init in init_test_ga:
  evaluate_test_size(
    tam=tam_init
  ) 
