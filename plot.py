from matplotlib import pyplot as plt

def generate_ga_plot(
  index: list, 
  picks: list,
  size_ga_init: int) -> None:

  plt.plot(index, picks)
  plt.title(f'GA Fitness. Población: {size_ga_init}')
  plt.xlabel('Época')
  plt.ylabel('Fitness')
  plt.savefig(f'GA_pop_{size_ga_init}.png')

def generate_val_size_plot(
  index_tam_val: list,
  def_hip: list,
  hip_ga: list,
  size_ga_init: int) -> None:

  plt.plot(index_tam_val, def_hip, label='SVC(A) Default')
  plt.plot(index_tam_val, hip_ga, label='SVC(B) Hiper-GA')
  plt.title(f'Validación de df con GA: {size_ga_init}')
  plt.xlabel('Tamaño de Lote')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig(f'GA_val_size_{size_ga_init}.jpg')
