import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
scaler = StandardScaler()
x = pd.read_csv('data/artificial_train.data.txt', sep=' ').iloc[:, :-1]
x = scaler.fit_transform(x)
y = pd.read_csv('data/artificial_train.labels.txt', sep=' ').to_numpy().ravel()

def accuracy_features_lasso(C: float) ->dict:
    model = LogisticRegression(penalty='l2', solver='saga', C=C)
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    model.fit(X_train, y_train)
    print(C)
    return {'c': C,
        'accuracy': (model.predict(X_test) == np.array(y_test).reshape(-1)).mean(),
            'n_features': np.sum(np.abs(model.coef_) < 1e-10)}


assessment = pd.DataFrame([accuracy_features_lasso(c) for c in 10**np.arange(-13, -10, 0.1)])
plt.figure()
plt.plot(assessment['c'], assessment['accuracy'])
plt.figure()
plt.plot(assessment['c'], assessment['n_features'])