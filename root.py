import os
import random
import numpy as np 
import pandas as pd
import matplotlib as plt

from pandas import DataFrame, array
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

""" Parámetros iniciales para SVC """

regularization: array = [1, 4, 8, 16, 32, 64, 128, 256]
kernel: array = ['linear', 'rbf', 'poly', 'sigmoid']
degree: array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
gama: array = ['scale', 'auto']
probability: array = [True, False] 

""" Parámetros iniciales para la selección de datos """

test_size: array = [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34]
random_state: array = list(range(1, 120))
sufffle: array = [True, False]

""" Funciones generadoras de configuraciones """

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

def generateChildrenConfig(hiper: list) -> object:
  return {
    'C': hiper[0], 'kernel': hiper[1],
    'degree': hiper[2], 'gamma': hiper[3],
    'probability': hiper[4]
  }

""" Clase IndividualSVCConfig """

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

""" Generar Población """

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

""" Obtener métricas del Modelo """

def getAccuracyForModel(y_test: np.ndarray, y_pred: np.ndarray) -> float:
  return float(metrics.accuracy_score(y_test, y_pred))










df_review = pd.read_csv('IMDB Dataset.csv')

def geneticAlgorithmForSVCConfig(df: DataFrame, nPop: int, epochs) -> None:
  pass