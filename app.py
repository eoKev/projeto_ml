from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html', classifiers=classificadores)

@app.route('/classifier_result', methods=['POST'])
def classifier_result():
    classificador_selecionado = request.form.get('classifier')
    clf = classificadores[classificador_selecionado]

    clf.fit(X_train_scaled_wine, y_train_wine)
    y_pred_wine = clf.predict(X_test_scaled_wine)

    acc_wine = accuracy_score(y_test_wine, y_pred_wine)
    f1_wine = f1_score(y_test_wine, y_pred_wine, average='macro')
    cm_wine = confusion_matrix(y_test_wine, y_pred_wine)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_wine, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Valor Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão para {}'.format(classificador_selecionado))

    img_confusion_matrix = io.BytesIO()
    FigureCanvas(plt.gcf()).print_png(img_confusion_matrix)
    img_confusion_matrix.seek(0)
    confusion_matrix_url = base64.b64encode(img_confusion_matrix.getvalue()).decode()

    plt.close()

    return render_template('result.html', classifier=classificador_selecionado, accuracy=acc_wine, f1_score=f1_wine,
                           confusion_matrix_url=confusion_matrix_url)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
