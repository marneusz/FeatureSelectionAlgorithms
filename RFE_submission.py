import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

for dataset in ["artificial"]:
    x = pd.read_csv(f"~/Python Projects/AML Project 2/AML_Project_2/data/{dataset}_train.data.txt", sep=" ").iloc[:, :-1]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = pd.read_csv(f"~/Python Projects/AML Project 2/AML_Project_2/data/{dataset}_train.labels.txt", sep=" ").to_numpy().ravel()
    y = (y + 1) // 2

    features_numbers = [ 28,  48,  64, 105, 128, 153, 241, 281, 318, 336, 338, 378, 433,
        442, 451, 453, 455, 472, 475, 493]
    x = x[:, features_numbers]
    model_transformed = RandomForestClassifier()
    model_transformed.fit(x, y)

    x_test = pd.read_csv(f"~/Python Projects/AML Project 2/AML_Project_2/data/{dataset}_valid.data.txt", sep=" ").iloc[:, :-1]
    x_test = scaler.transform(x_test)
    x_test_transformed = x_test[:, features_numbers]

    print(x_test.shape, x_test_transformed.shape)
    print(model_transformed.classes_)
    pd.DataFrame(model_transformed.predict_proba(x_test_transformed)[:, 1]).to_csv(
        f"results/{dataset}_valid.labels.txt", index=False, header="PAUPAC"
    )
    
