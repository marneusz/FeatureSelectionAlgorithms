import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

scaler = StandardScaler()
x = pd.read_csv("data/digits_train.data.txt", sep=" ").iloc[:, :-1]
x = scaler.fit_transform(x)
y = pd.read_csv("data/digits_train.labels.txt", sep=" ").to_numpy().ravel()
y = (y + 1) // 2
X_train, X_test, y_train, y_test = train_test_split(x, y)


def accuracy_features_lasso(C: float) -> dict:
    model = LogisticRegression(penalty="l1", solver="saga", C=C)
    model.fit(X_train, y_train)
    return {
        "c": C,
        "accuracy": (model.predict(X_test).ravel() == y_test.ravel()).mean(),
        "n_features": np.sum(np.abs(model.coef_) < 1e-10),
    }


assessment = pd.DataFrame(
    [accuracy_features_lasso(c) for c in 10.0 ** np.arange(-4, 2, 1)]
)

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("C")
ax1.set_xscale("log")
ax1.plot((assessment["c"]), assessment["accuracy"], color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim((0, 1))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_xscale("log")
color = "tab:blue"
ax2.set_ylabel("Number of features", color=color)
ax1.set_ylabel("Accuracy", color="tab:red")  # we already handled the x-label with ax1
ax2.plot((assessment["c"]), assessment["n_features"], color=color)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim((0, 5150))
plt.title("Lasso regression for digits dataset.")
plt.show()


## Validation. I am choosing C=100 for the model

validation = pd.read_csv("data/digits_valid.data.txt", sep=" ").iloc[:, :-1]
validation = scaler.transform(validation)
model = LogisticRegression(penalty="l1", solver="saga", C=100.0)
model.fit(x, y)
pd.DataFrame(model.predict(validation)).to_csv(
    "results/artificial_valid.labels.txt", index=False, header=None
)
