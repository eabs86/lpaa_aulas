from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


# Gerando dados de exemplo
X, y = make_moons(n_samples=100, noise=0.25, random_state=42)

# Dividindo os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
classifier = DecisionTreeClassifier(max_depth=5)
classifier.fit(X_train, y_train)

# Visualizando a zona de separação
plot_decision_regions(X, y, clf=classifier, legend=2)
plt.title('Classificação com Árvore de Decisão')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper left')
plt.show()
