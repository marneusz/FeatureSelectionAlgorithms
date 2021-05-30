import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.preprocessing import StandardScaler

for dataset in ["artificial", "digits"]:
    x = pd.read_csv(f"~/Python Projects/AML Project 2/AML_Project_2/data/{dataset}_train.data.txt", sep=" ").iloc[:, :-1]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = pd.read_csv(f"~/Python Projects/AML Project 2/AML_Project_2/data/{dataset}_train.labels.txt", sep=" ").to_numpy().ravel()
    y = (y + 1) // 2
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    model = RandomForestClassifier()
    fs = BorutaPy(model, n_estimators='auto', verbose=2)
    fs.fit(np.array(X_train), np.array(y_train))
    X_filtered = fs.transform(np.array(X_train))
    model_filtered = RandomForestClassifier()
    model_filtered.fit(X_filtered, y_train)
    acc = (model_filtered.predict(fs.transform(np.array(X_test))).ravel() == y_test.ravel()).mean()
    important = fs.support_
    print(
        f"{dataset}: Accuracy {round(acc,3)}, Number of important features: {important.sum()} / {len(important)}"
    )
    print(
        f"{dataset}: Selected variables {np.where(fs.support_ == True)[0]}"
    )
