import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Carregando os dados
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
import os
import seaborn as sns
from itertools import combinations
from matplotlib import pyplot as plt

# Obtendo o diretório atual do script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construindo o caminho para o arquivo de dados
data_path = os.path.join(script_dir, 'dados', 'iris.data')
df = pd.read_csv(data_path, names=column_names)

def plot_decision_boundaries(X, y, clf, feature_names, class_names, resolution=0.02):
    """
    Gera gráficos mostrando as regiões de decisão do modelo para cada par de características.
    """
    # Configurações de estilo
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 8)
    
    # Cores para cada classe
    colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF']
    
    # Gerando gráficos para cada par de características
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            # Pegando apenas as duas características atuais
            X_pair = X[:, [i, j]]
            
            # Criando o classificador apenas com as duas características
            clf_pair = DecisionTreeClassifier(random_state=42, max_depth=2)
            clf_pair.fit(X_pair, y)
            
            # Configurando o gráfico
            plt.figure()
            
            # Definindo os limites do gráfico
            x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
            y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
            
            # Criando a grade de pontos para o gráfico
            xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                               np.arange(y_min, y_max, resolution))
            
            # Fazendo previsões para cada ponto da grade
            Z = clf_pair.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = np.array([list(class_names).index(z) for z in Z])
            Z = Z.reshape(xx.shape)
            
            # Plotando as regiões de decisão
            plt.contourf(xx, yy, Z, alpha=0.3, levels=len(class_names)-1, colors=colors)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            
            # Plotando os pontos de dados
            for idx, class_name in enumerate(class_names):
                plt.scatter(X_pair[y == class_name, 0], 
                           X_pair[y == class_name, 1],
                           c=colors[idx], 
                           label=class_name,
                           edgecolor='black', 
                           s=80)
            
            # Adicionando rótulos e título
            plt.xlabel(feature_names[i].replace('_', ' ').title())
            plt.ylabel(feature_names[j].replace('_', ' ').title())
            plt.title(f'Regiões de Decisão: {feature_names[i].replace("_", " ").title()} vs {feature_names[j].replace("_", " ").title()}')
            
            # Adicionando legenda
            plt.legend(title='Espécie', title_fontsize='large')
            
            # Adicionando grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Ajustando layout
            plt.tight_layout()
            
            # Salvando a figura
            plt.savefig(f'decision_boundary_{feature_names[i]}_vs_{feature_names[j]}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print("\nGráficos das regiões de decisão salvos como 'decision_boundary_[característica1]_vs_[característica2].png'")

def plot_iris_features(df):
    """
    Gera gráficos de dispersão individuais para cada par de características.
    Cada gráfico é salvo separadamente para melhor visualização.
    """
    # Configurações de estilo
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 8)
    
    # Lista de características
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Cores para cada espécie
    palette = {'Iris-setosa': 'blue', 'Iris-versicolor': 'green', 'Iris-virginica': 'red'}
    
    # Gerando gráficos para cada par de características
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            x_feature = features[i]
            y_feature = features[j]
            
            # Criando a figura
            plt.figure()
            
            # Plotando cada espécie separadamente para melhor controle
            for species, color in palette.items():
                species_data = df[df['class'] == species]
                plt.scatter(species_data[x_feature], species_data[y_feature], 
                            color=color, label=species, alpha=0.7, s=80, edgecolor='w')
            
            # Adicionando rótulos e título
            plt.xlabel(x_feature.replace('_', ' ').title())
            plt.ylabel(y_feature.replace('_', ' ').title())
            plt.title(f'{x_feature.replace("_", " ").title()} vs {y_feature.replace("_", " ").title()}')
            
            # Adicionando legenda
            plt.legend(title='Espécie', title_fontsize='large')
            
            # Adicionando grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Ajustando layout
            plt.tight_layout()
            
            # Salvando a figura
            plt.savefig(f'iris_{x_feature}_vs_{y_feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("Gráficos de dispersão salvos como 'iris_[característica1]_vs_[característica2].png'")

# Gerando as visualizações dos dados brutos
plot_iris_features(df.copy())

# Separando features e target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Criando e treinando o modelo para visualização das regiões de decisão
clf_vis = DecisionTreeClassifier(random_state=42, max_depth=2)
clf_vis.fit(X, y)

# Obtendo os nomes das características e classes
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
class_names = np.unique(y)

# Gerando as visualizações das regiões de decisão
plot_decision_boundaries(X, y, clf_vis, feature_names, class_names)

# i) Treinamento e Teste com diferentes proporções
def train_test_evaluation(test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Criando e treinando o modelo
    clf = DecisionTreeClassifier(random_state=42, max_depth=2)
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

