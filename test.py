import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df_review = pd.read_csv('IMDB Dataset.csv')

# df_review['sentiment'].replace(['negative', 'positive'], [0, 1], inplace = True)


# df_positive = df_review.loc[df_review['sentiment'] == 1]
# df_negative = df_review.loc[df_review['sentiment'] == 0]

df_positive = df_review.loc[df_review['sentiment'] == 'positive']
df_negative = df_review.loc[df_review['sentiment'] == 'negative']

df_positive = df_positive[:1000]
df_negative = df_negative[:1000]

frames = [df_positive, df_negative]
df_final = pd.concat(frames)


X = df_final['review']
y = df_final['sentiment']

print(df_final)

"""
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)

test_x_vector = tfidf.transform(test_x)

"""


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 56, shuffle = True)

tfid = TfidfVectorizer(stop_words = 'english')

X_train = tfid.fit_transform(X_train)
X_test = tfid.transform(X_test)



from sklearn.svm import SVC

print(f'Executing SVC')

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

print(svc.predict(tfid.transform(['I did not like this movie at all I gave this movie away'])))





