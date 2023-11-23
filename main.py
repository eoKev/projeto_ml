import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()

df_wine = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])
X_wine = df_wine.drop('target', axis=1)
y_wine = df_wine['target']

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, random_state=1)
scaler_wine = StandardScaler()
X_train_scaled_wine = scaler_wine.fit_transform(X_train_wine)
X_test_scaled_wine = scaler_wine.transform(X_test_wine)

classificadores = {
    'Perceptron': Perceptron(penalty='l1', alpha=0.01, max_iter=15),
    'SVM': SVC(C=1.0, kernel='rbf', gamma='scale'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=2, criterion='gini'),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
}

for classificador, clf in classificadores.items():
    clf.fit(X_train_scaled_wine, y_train_wine)
    y_pred_wine = clf.predict(X_test_scaled_wine)

    acc_wine = accuracy_score(y_test_wine, y_pred_wine)
    f1_wine = f1_score(y_test_wine, y_pred_wine, average='macro')

    print(f'{classificador}:')
    print(f'Acurácia: {acc_wine:.2f}')
    print(f'Macro Average F1-Score: {f1_wine:.2f}\n')
