from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

tfid = TfidfVectorizer(stop_words='english')

def get_accuracy(
  y_test: np.ndarray, 
  y_pred: np.ndarray
  ) -> float:
  
  return float(metrics.accuracy_score(y_test, y_pred))

def evaluate_population(
  populat: list,
  X, y) -> None:

  x_conf: object
  x_split: object

  for indi in populat:
    x_conf = indi.get_config()
    x_split = indi.get_split()

    X_train, X_test, y_train, y_test = train_test_split(X, y, **x_split)

    X_train = tfid.fit_transform(X_train)
    X_test = tfid.transform(X_test)

    svc = SVC(**x_conf)
    svc.fit(X_train, y_train)
    predicted = svc.predict(X_test)
    indi.set_fitness(get_accuracy(y_test=y_test, y_pred=predicted))
