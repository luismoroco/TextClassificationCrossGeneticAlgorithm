from params import gen_config, gen_split, gen_config_from, gen_split_from
from individual import Individuo
import random

def gen_population(
  nPop: int
  ) -> list:

  population: list = []
  print(f'Generando una Población de {nPop} individuos')

  config: object
  splitConf: object
  fitnessDefault: float = 0

  for _ in range(int(nPop)):
    config = gen_config()
    splitConf = gen_split()
    individuo: Individuo = Individuo(
      f=fitnessDefault, 
      conf=config, 
      split=splitConf
    )
    population.append(individuo)

  return population

def crossover(
  fath: Individuo, 
  moth: Individuo
  ) -> Individuo:

  fath_cong: list = list(fath.get_config().values())
  moth_conf: list = list(moth.get_config().values())
  point_cross: int = random.randint(0, len(fath_cong)-1)
  fsplit: list = list(fath.get_split().values())
  msplit: list = list(moth.get_split().values())

  for index in range(point_cross, len(fath_cong)):
    fath_cong[index] = moth_conf[index]

  for index in range(point_cross, len(fsplit)):
    fsplit[index] = msplit[index]

  newConf: object = gen_config_from(fath_cong)
  newCOnfSplit: object = gen_split_from(fsplit)

  fitnessDefault: float = 0
  children: Individuo = Individuo(
    f=fitnessDefault, 
    conf=newConf, 
    split=newCOnfSplit
  )

  return children

def select_parents_withouth_bias(
  population: list) -> list:
  
  total_fitness: float = 0
  fitness_scale: list = []

  for index, indi in enumerate(population):
    total_fitness += indi.get_fitness()
    if index == 0:
      fitness_scale.append(indi.get_fitness())
    else:
      fitness_scale.append(indi.get_fitness() + fitness_scale[index-1])

  mating_pool: list = []
  number_parents = len(population)
  fitness_step = total_fitness/number_parents

  random_offset = random.uniform(0, fitness_step)

  current_fitness_pointer: float = random_offset
  last_fitness_scale_position: int = 0
  for index in range(len(population)):
    for fitness_scale_position in range(last_fitness_scale_position, len(fitness_scale)):
      if fitness_scale[fitness_scale_position] >= current_fitness_pointer:
        mating_pool.append(population[fitness_scale_position])
        last_fitness_scale_position = fitness_scale_position
        break
    current_fitness_pointer += fitness_step
  
  return mating_pool

