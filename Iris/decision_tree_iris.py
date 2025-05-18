import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Carregando os dados
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
import os
# Obtendo o diretório atual do script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construindo o caminho para o arquivo de dados
data_path = os.path.join(script_dir, 'dados', 'iris.data')
df = pd.read_csv(data_path, names=column_names)

# Separando features e target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# i) Treinamento e Teste com diferentes proporções
def train_test_evaluation(test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Criando e treinando o modelo
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Fazendo previsões
    y_pred = clf.predict(X_test)
    
    # Avaliando o modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAvaliação com {int((1-test_size)*100)}% treino e {int(test_size*100)}% teste:")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    
    # Plotando a árvore de decisão
    plt.figure(figsize=(20,10))
    plot_tree(clf, feature_names=column_names[:-1], class_names=df['class'].unique(), filled=True, rounded=True)
    plt.title(f"Árvore de Decisão - {int((1-test_size)*100)}% Treino / {int(test_size*100)}% Teste")
    plt.savefig(f'decision_tree_{int((1-test_size)*100)}_{int(test_size*100)}.png')
    plt.close()
    
    return accuracy

# Avaliando diferentes proporções de treino/teste
proportions = [0.2, 0.25, 0.3, 0.4]
for prop in proportions:
    train_test_evaluation(test_size=prop)

# ii) 10-fold Cross Validation
def cross_validation_evaluation():
    clf = DecisionTreeClassifier(random_state=42)
    
    # 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
    
    print("\n10-Fold Cross Validation Results:")
    print(f"Acurácia Média: {cv_scores.mean():.4f}")
    print(f"Desvio Padrão: {cv_scores.std():.4f}")
    print(f"Acurácia de cada fold: {cv_scores}")
    
    # Treinando o modelo final com todos os dados para visualização
    clf.fit(X, y)
    plt.figure(figsize=(20,10))
    plot_tree(clf, feature_names=column_names[:-1], class_names=df['class'].unique(), filled=True, rounded=True)
    plt.title("Árvore de Decisão - 10-Fold Cross Validation")
    plt.savefig('decision_tree_10fold_cv.png')
    plt.close()

# Executando a validação cruzada
cross_validation_evaluation()

