import random
from pandas import array
from sklearn import metrics
import numpy as np

"""
Los Hiperparámetros(Configuraciones) serán generados de forma aleatoria, tomandolos
de un banco de posibles datos. Tener un gran espacio de búsqueda es característico de 
los algoritmos genéticos
"""

""" Parámetros iniciales """ 

regularization: array = [1, 4, 8, 16, 32, 64, 128, 256]
kernel: array = ['linear', 'rbf', 'poly', 'sigmoid']
degree: array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
gama: array = ['scale', 'auto']
probability: array = [True, False] 

""" Parámetros para selección de datos """ 

test_size: array = [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39]
random_state: array = list(range(1, 120))
sufffle: array = [True, False]

""" Funciones para obtener configuraciones """

def generateConfigForSVC() -> object:
  try:
    return {
      'C': random.choice(regularization), 'kernel': random.choice(kernel),
      'degree': random.choice(degree), 'gamma': random.choice(gama),
      'probability': random.choice(probability)
    }
  except:
    return generateConfigForSVC()

def generateConfigForSplitData() -> object:
  try:
    return {
      'test_size': random.choice(test_size), 'random_state': random.choice(random_state),
      'shuffle': random.choice(sufffle)
    }
  except:
    return generateConfigForSplitData()

def generateChildrenConfig(hiper: list) -> object:
  try:
    return {
      'C': hiper[0], 'kernel': hiper[1],
      'degree': hiper[2], 'gamma': hiper[3],
      'probability': hiper[4]
    }
  except:
    return generateChildrenConfig(hiper = hiper)

def getAccuracyForModel(
  y_test: np.ndarray, 
  y_pred: np.ndarray) -> float:
  try:
    return float(metrics.accuracy_score(y_test, y_pred))
  except:
    raise ValueError('Error in getAccuracyForModel')

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