from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Gerando 100 dados de exemplo
np.random.seed(0) # Para garantir a reprodutibilidade
X = 2.5 * np.random.randn(100, 1) + 1.5   # 100 pontos de dados para variável independente
y = 2 + 0.5 * X.flatten() + np.random.randn(100) # Variável dependente com ruído

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Previsões
y_pred = modelo.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)

# Visualizar a linha de melhor ajuste
plt.scatter(X, y, color='black')
plt.plot(X, modelo.predict(X), color='blue', linewidth=3)
plt.title('Regressão Linear')
plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.show(), mse