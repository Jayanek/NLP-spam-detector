import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

data = pd.read_csv("spambase.data", sep=",", header=None)

Shuffle_data = shuffle(data)

inputs = Shuffle_data.iloc[:, :57]
labels = Shuffle_data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3)

model_01 = MultinomialNB()
model_01.fit(x_train, y_train)

score_01 = model_01.score(x_test, y_test)

model_02 = AdaBoostClassifier()

model_02.fit(x_train, y_train)

score_02 = model_02.score(x_test, y_test)

print("Using NB : ", score_01, " Using AdaBoost : ", score_02)

prediction01 = model_01.predict(x_test)
prediction02 = model_02.predict(x_test)


print("NB : ", prediction01[i], " Ada: ", prediction02[i], " Actual", y_test[i])
