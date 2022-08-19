import random
from data import regularization, kernel, degree, gama, probability, test_size, random_state, sufffle

def gen_config(
  ) -> object:
  return {
    'C': random.choice(regularization), 
    'kernel': random.choice(kernel),
    'degree': random.choice(degree), 
    'gamma': random.choice(gama),
    'probability': random.choice(probability)
  }

def gen_split(
  ) -> object:
  return {
    'test_size': random.choice(test_size), 
    'random_state': random.choice(random_state),
    'shuffle': random.choice(sufffle)
  }

def gen_config_from(
  hiper: list
  ) -> object:
  return {
    'C': hiper[0], 
    'kernel': hiper[1],
    'degree': hiper[2], 
    'gamma': hiper[3],
    'probability': hiper[4]
}

def gen_split_from(
  hiper: list
  ) -> object:
  return {
    'test_size': hiper[0], 
    'random_state': hiper[1],
    'shuffle': hiper[2]
  }

def get_mutation(
  x: int
  ) -> any:
  if x == 0:
    return random.choice(regularization)
  if x == 1:
    return random.choice(kernel)
  if x == 2:
    return random.choice(degree)
  if x == 3:
    return random.choice(gama)
  if x == 4:
    return random.choice(probability)
