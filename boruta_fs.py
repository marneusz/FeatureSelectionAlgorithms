import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from boruta import BorutaPy

for dataset in ["digits"]:
    x = pd.read_csv(f"~/Python Projects/AML Project 2/AML_Project_2/data/{dataset}_train.data.txt", sep=" ").iloc[:, :-1]
    y = pd.read_csv(f"~/Python Projects/AML Project 2/AML_Project_2/data/{dataset}_train.labels.txt", sep=" ").to_numpy().ravel()
    y = (y + 1) // 2
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    model = RandomForestClassifier(n_jobs=-1,
                                   class_weight='balanced',
                                   max_depth=5)
    fs = BorutaPy(model, n_estimators='auto', verbose=2)
    fs.fit(X_train, y_train)
    X_filtered = fs.transform(X_train)
    model_filtered = RandomForestClassifier(n_jobs=-1,
                                            class_weight='balanced',
                                            max_depth=5)
    acc = (model_filtered.predict(fs.transform(X_test)).ravel() == y_test.ravel()).mean()
    important = fs.support_
    print(
        f"{dataset[0].upper()}: Accuracy {round(acc,3)}, Number of important features: {important.sum()} / {len(important)}"
    )
