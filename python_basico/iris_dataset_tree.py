from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Carregando um conjunto de dados de exemplo
iris = load_iris()
X, y = iris.data, iris.target

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo de árvore de decisão
classifier = DecisionTreeClassifier(max_depth=4)
classifier.fit(X_train, y_train)

# Plotando o diagrama da árvore
plt.figure(figsize=(20,10))
plot_tree(classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.show()
