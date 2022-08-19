from pandas import array

regularization: array = [1, 4, 8, 16, 32, 64, 128, 256]
kernel: array = ['linear', 'rbf', 'poly', 'sigmoid']
degree: array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
gama: array = ['scale', 'auto']
probability: array = [True, False]

test_size: array = [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23,
                    0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34]
random_state: array = list(range(1, 120))
sufffle: array = [True, False]
