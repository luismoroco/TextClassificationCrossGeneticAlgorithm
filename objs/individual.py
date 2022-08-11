import os 

"""
  La clase IndiviudalSVCConfig contiene la información de nuestro modelo.
  Contiene 3 argumentos: fitness(Accuracy de cada modelo entrenado), 
  config(Hiperparámetros de la SVC generado aleatoriamente) y 
  configSplit(Configuración de la selección de datos) 
"""

class IndiviudalSVCConfig:
  fitness: float 
  config: object
  configSplit: object

  def __init__(
    self, 
    f: float, 
    conf: object, 
    split: object) -> None:
    try:
      self.fitness = f
      self.config = conf
      self.configSplit = split
    except:
      raise ValueError('Error __init__')

  def getFitness(self) -> float:
    try:
      return float(self.fitness)
    except:
      raise ValueError('Error getFitness')

  def getConfig(self) -> object:
    try:
      return (self.config)
    except:
      raise ValueError('Error getConfig')
  
  def setFitness(self, fit: float) -> None:
    try:
      self.fitness = fit
    except:
      raise ValueError('Error setFitness')

  def getConfigSplit(self) -> object:
    try:
      return (self.configSplit) 
    except: 
      raise ValueError('Error getConfigSplit')

  def setConfigSplit(self, split: object) -> None:
    try:
      self.configSplit = split 
    except: 
      raise ValueError('Error setConfigSplit')
