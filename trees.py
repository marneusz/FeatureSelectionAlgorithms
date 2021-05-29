import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

for dataset in ["artificial", "digits"]:
    x = pd.read_csv(f"data/{dataset}_train.data.txt", sep=" ").iloc[:, :-1]
    y = pd.read_csv(f"data/{dataset}_train.labels.txt", sep=" ").to_numpy().ravel()
    y = (y + 1) // 2
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    model = RandomForestClassifier()
    sel = SelectFromModel(model)
    sel.fit(X_train, y_train)
    model.fit(X_train, y_train)
    acc = (model.predict(X_test).ravel() == y_test.ravel()).mean()
    important = sel.get_support()

    print(
        f"{dataset[0].upper() + dataset[1:]}: Accuracy {round(acc,3)}, Number of important features: {important.sum()} / {len(important)}"
    )
