from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Gerando dados de exemplo
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.15, size=X.shape[0])

# Dividindo os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
regressor = DecisionTreeRegressor(max_depth=5)
regressor.fit(X_train, y_train)

# Previsões
X_grid = np.arange(X.min(), X.max(), 0.01)[:, np.newaxis]
y_pred = regressor.predict(X_grid)

# Plotando
plt.scatter(X, y, color='red', label='Dados')
plt.plot(X_grid, y_pred, color='blue', label='Modelo de Regressão')
plt.title('Regressão com Árvore de Decisão')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
