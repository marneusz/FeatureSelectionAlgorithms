import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

for dataset in ["digits"]:
    x = pd.read_csv(f"~/Python Projects/AML Project 2/AML_Project_2/data/{dataset}_train.data.txt", sep=" ").iloc[:, :-1]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = pd.read_csv(f"~/Python Projects/AML Project 2/AML_Project_2/data/{dataset}_train.labels.txt", sep=" ").to_numpy().ravel()
    y = (y + 1) // 2
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    model = RandomForestClassifier(random_state=47)
    rfecv = RFECV(estimator=model, step=5, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(X_train, y_train)
    print('Optimal number of features: {}'.format(rfecv.n_features_))

    plt.figure(figsize=(16, 9))
    plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
    plt.savefig(f"forward_cv_plot_{dataset}.png")

    print(np.where(rfecv.support_ == True)[0])
    X_transformed = X_train[:, np.where(rfecv.support_ == True)[0]]
    model_transformed = RandomForestClassifier()
    model_transformed.fit(X_transformed, y_train)

    X_test_transformed = X_test[:, np.where(rfecv.support_ == True)[0]]
    acc = (model_transformed.predict(X_test_transformed).ravel() == y_test.ravel()).mean()
    important = rfecv.support_
    print(
        f"{dataset}: Accuracy {round(acc,3)}, Number of important features: {important.sum()} / {len(important)}"
    )