from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Reajustando os parâmetros do conjunto de dados para introduzir mais sobreposição e, portanto, mais dificuldade
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.2, random_state=42)

# Dividir os dados em conjunto de treinamento e teste novamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão logística novamente
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Previsões e avaliação do modelo novamente
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Visualização com fronteira de decisão
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='winter')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', marker='o', edgecolor='black', s=20)
plt.title('Regressão Logística - Fronteira de Decisão')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

accuracy

