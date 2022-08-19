import random
from params import get_mutation, gen_config_from

class Individuo:
  fitness: float
  config: object
  configSplit: object

  def __init__(
    self, 
    f: float, 
    conf: object = {}, 
    split: object = {}) -> None:
      self.fitness = f
      self.config = conf
      self.configSplit = split

  def get_fitness(
    self) -> float:
    return float(self.fitness)

  def get_config(
    self) -> object:
    return (self.config)

  def set_fitness(
    self, 
    fit: float) -> None:
    self.fitness = fit

  def get_split(
    self) -> object:
    return (self.configSplit)

  def set_split(
    self, 
    split: object) -> None:
    self.configSplit = split

  def mutate(
    self) -> None:
    current_config: list = list(self.get_config().values())
    pos_change: int = random.randint(0, len(current_config)-1)
    current_config[pos_change] = get_mutation(x=pos_change)
    mutate_conf: object = gen_config_from(current_config)
    self.config = mutate_conf
