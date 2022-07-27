from operator import mod
import os
from pyexpat import model 
import random
import numpy as np 
import pandas as pd
import matplotlib as plt
from pandas import array
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Parámetros iniciales 

regularization: array = [1, 4, 8, 16, 32, 64, 128, 256]
kernel: array = ['linear', 'rbf', 'poly', 'sigmoid']
degree: array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
gama: array = ['scale', 'auto']
probability: array = [True, False] 

# Parámetros para selección de datos 

test_size: array = [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39]
random_state: array = list(range(1, 120))
sufffle: array = [True, False]

def generateConfigForSVC() -> object:
  return {
    'C': random.choice(regularization), 'kernel': random.choice(kernel),
    'degree': random.choice(degree), 'gamma': random.choice(gama),
    'probability': random.choice(probability)
  }

def generateConfigForSplitData() -> object:
  return {
    'test_size': random.choice(test_size), 'random_state': random.choice(random_state),
    'shuffle': random.choice(sufffle)
  }

# Clase Individuo 

class IndiviudalSVCConfig:
  fitness: float 
  config: object
  configSplit: object

  def __init__(self, f: float, conf: object, split: object) -> None:
    self.fitness = f
    self.config = conf
    self.configSplit = split

  def getFitness(self) -> float:
    return float(self.fitness)

  def getConfig(self) -> object:
    return (self.config)
  
  def setFitness(self, fit: float) -> None:
    self.fitness = fit

  def getConfigSplit(self) -> object:
    return (self.configSplit) 

  def setConfigSplit(self, split: object) -> None:
    self.configSplit = split 

# Crear Población 

def generatePopulation(nPop: int) -> list:
  population: list = []
  print(f'Generando una Población de {nPop} individuos')
  
  config: object
  splitConf: object
  fitnessDefault: float = 0
  for _ in range(int(nPop)):
    config = generateConfigForSVC()
    splitConf = generateConfigForSplitData()
    individuo: IndiviudalSVCConfig = IndiviudalSVCConfig(
      f = fitnessDefault, conf = config, split = splitConf
    )
    population.append(individuo)

  return population

# Cargar Dataset de Clasificación

df = pd.read_csv('diabetes.csv', delimiter = ',')

etiquet = df['Outcome']
features = df.iloc[:,0:8]

X = features 
y = np.ravel(etiquet)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 56, shuffle = True) 

"""
X_train, X_test, y_train, y_test = train_test_split(X, y, **generateConfigForSplitData()) 

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
"""

# Generar Métricas 

def getAccuracyForModel(y_test: np.ndarray, y_pred: np.ndarray) -> float:
  return float(metrics.accuracy_score(y_test, y_pred))

# Evaluar Población

def evaluatePopulation(populat: list) -> None:
  print(f'Evaluando a los individuos tamaño {len(populat)}')
  count: int = 0
  xConf: object 
  splitC: object
  for indi in populat:
    print(f'Evaluando individuo {count}')
    xConf = indi.getConfig() 
    splitC = indi.getConfigSplit()
    X_train, X_test, y_train, y_test = train_test_split(X, y, **splitC) 
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    svc = SVC(**xConf)
    svc.fit(X_train, y_train)
    predicted = svc.predict(X_test)
    indi.setFitness(getAccuracyForModel(y_test = y_test, y_pred = predicted))
    count += 1

def mutationForIndivudualConfigSVC(fath: IndiviudalSVCConfig) -> IndiviudalSVCConfig: 
  pass


def crossoverForIndivudualConfigSVC(fath: IndiviudalSVCConfig, moth: IndiviudalSVCConfig) -> IndiviudalSVCConfig:
  pass


def geneticAlgorithmInit(nPop: int) -> IndiviudalSVCConfig:
  initialPopulation: list = generatePopulation()
  
  pass









# Pruebas 

population: list = generatePopulation(10)
evaluatePopulation(population)

count: int = 0
population.sort(key = lambda fit: fit.fitness, reverse = True)

for it in population:
  print(f'Individuo {count} : Fitness = {it.getFitness()}')
  count += 1












