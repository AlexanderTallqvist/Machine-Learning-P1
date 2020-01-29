import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

patients = pd.read_csv("D:\Xyli\GitHub\Machine-learning-project1\data3.csv",
                       encoding='latin1',
                       error_bad_lines=False,
                       sep=";",
                       skiprows=1,
                       header=None,
                       names=["Reason", "Gender", "Age", "Mobility", "Distance", "Participation"]
                       )

df = pd.DataFrame(patients)

X = df[['Mobility', 'Distance']]
y = df['Participation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

print(X_test)
print(y_pred)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print(confusion_matrix)

sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

plt.show()
