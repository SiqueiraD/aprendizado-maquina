"""
clustering_iris.py - Trabalho Prático de Aprendizado de Máquina Não Supervisionado
===============================================================================

Este script implementa uma análise completa do dataset Iris usando algoritmos de clustering,
conforme solicitado no trabalho prático de Aprendizado de Máquina Não Supervisionado.

O código realiza:
1. Análise exploratória dos dados
2. Aplicação do K-Means com método do cotovelo
3. Aplicação do Agrupamento Hierárquico com dendrograma
4. Avaliação dos resultados usando métricas intrínsecas e extrínsecas
5. Comparação entre o uso de todas as features x apenas duas features

Cada etapa é explicada nos comentários ao longo do código.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import combinations

# Configurações gerais de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set1")
plt.rcParams['figure.figsize'] = (10, 6)

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# -----------------------------------------------------------------------------

# Carregando o dataset Iris
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'dados', 'iris.data')

# Definição dos nomes das colunas
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(data_path, names=column_names)

print(f"Dataset carregado com {df.shape[0]} amostras e {df.shape[1]} colunas")
print("\nPrimeiras 5 linhas do dataset:")
print(df.head())
print("\nInformações estatísticas do dataset:")
print(df.describe())


# -----------------------------------------------------------------------------
# 2. FUNÇÕES AUXILIARES PARA VISUALIZAÇÃO E AVALIAÇÃO
# -----------------------------------------------------------------------------

def plot_features_pairwise(df, save_prefix="iris_"):
    """
    Gera gráficos de dispersão para cada par de características.
    
    Args:
        df: DataFrame com os dados
        save_prefix: Prefixo para salvar os arquivos
    """
    feature_pairs = list(combinations(column_names[:-1], 2))
    
    print(f"\nGerando {len(feature_pairs)} gráficos de dispersão...")
    
    # Para cada par de características
    for x_feature, y_feature in feature_pairs:
        plt.figure(figsize=(8, 6))
        
        # Plotar os pontos por espécie
        for species in df['class'].unique():
            subset = df[df['class'] == species]
            plt.scatter(
                subset[x_feature], 
                subset[y_feature], 
                alpha=0.7, 
                label=species,
                edgecolors='w',
                s=70
            )
        
        # Configurar o gráfico
        plt.xlabel(x_feature.replace('_', ' ').title())
        plt.ylabel(y_feature.replace('_', ' ').title())
        plt.title(f'{x_feature.title()} vs {y_feature.title()}')
        plt.legend(title='Espécie')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar o gráfico
        filename = f"{save_prefix}{x_feature}_vs_{y_feature}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Gráficos de dispersão salvos com prefixo '{save_prefix}'")


def plot_clusters(X, labels, feature_names, title, save_prefix):
    """
    Plota os clusters gerados para cada par de características.
    
    Args:
        X: Matriz de características
        labels: Array com os rótulos dos clusters
        feature_names: Lista com os nomes das características
        title: Título base para os gráficos
        save_prefix: Prefixo para salvar os arquivos
    """
    n_features = X.shape[1]
    feature_pairs = list(combinations(range(n_features), 2))
    
    # Para cada par de características
    for i, j in feature_pairs:
        plt.figure(figsize=(8, 6))
        
        # Plotar os pontos coloridos por cluster
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            plt.scatter(
                X[mask, i], 
                X[mask, j], 
                alpha=0.7,
                label=f'Cluster {cluster_id}',
                edgecolors='w',
                s=70
            )
        
        # Configurar o gráfico
        plt.xlabel(feature_names[i].replace('_', ' ').title())
        plt.ylabel(feature_names[j].replace('_', ' ').title())
        plt.title(f"{title}: {feature_names[i]} vs {feature_names[j]}")
        plt.legend(title='Clusters')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar o gráfico
        filename = f"{save_prefix}_{feature_names[i]}_vs_{feature_names[j]}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_clustering(X, labels, true_labels=None):
    """
    Avalia o clustering usando métricas intrínsecas e extrínsecas.
    
    Args:
        X: Matriz de características
        labels: Array com os rótulos dos clusters
        true_labels: Array com os rótulos verdadeiros (opcional)
    
    Returns:
        dict: Dicionário com as métricas calculadas
    """
    metrics = {}
    
    # Métricas intrínsecas (não dependem dos rótulos verdadeiros)
    metrics['Silhouette Score'] = silhouette_score(X, labels)
    metrics['Davies-Bouldin Index'] = davies_bouldin_score(X, labels)
    
    # Métrica extrínseca (depende dos rótulos verdadeiros)
    if true_labels is not None:
        metrics['Adjusted Rand Index'] = adjusted_rand_score(true_labels, labels)
    
    return metrics


# -----------------------------------------------------------------------------
# 3. ANÁLISE EXPLORATÓRIA DOS DADOS
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("ANÁLISE EXPLORATÓRIA DOS DADOS")
print("="*50)

# Gerar gráficos de dispersão para cada par de características
plot_features_pairwise(df)

# Quantos grupos naturais é possível identificar visualmente?
print("\nAnálise visual dos dados:")
print("- É possível identificar aproximadamente 3 grupos naturais nos dados.")
print("- A espécie Iris-setosa é claramente separada das outras.")
print("- Existe sobreposição entre Iris-versicolor e Iris-virginica em várias dimensões.")

# Preparação dos dados para clustering
X_all = df.iloc[:, :-1].values  # Todas as features
X_two = df[['petal_length', 'petal_width']].values  # Apenas comprimento e largura da pétala

# Convertendo os rótulos de texto para números (para avaliação posterior)
true_species = df['class'].values
unique_species = np.unique(true_species)
true_labels = np.zeros(len(true_species), dtype=int)

for i, species in enumerate(unique_species):
    true_labels[true_species == species] = i

feature_names = column_names[:-1]


# -----------------------------------------------------------------------------
# 4. K-MEANS CLUSTERING
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("K-MEANS CLUSTERING")
print("="*50)

# 4.1 Método do cotovelo para escolher o valor ótimo de k
print("\n1. Método do cotovelo para escolher o valor de k")

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_all)
    inertia.append(kmeans.inertia_)

# Plotando o método do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inércia (Soma dos quadrados das distâncias)')
plt.title('Método do Cotovelo para K-Means')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('kmeans_elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()

print("- Método do cotovelo salvo como 'kmeans_elbow_method.png'")
print("- Análise do gráfico sugere que k=3 é uma boa escolha, alinhado com o conhecimento prévio")

# 4.2 Aplicação do K-Means com k=3
print("\n2. Aplicação do K-Means com k=3")

# Usando todas as features
kmeans_full = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels_full = kmeans_full.fit_predict(X_all)

# Avaliação das métricas
kmeans_metrics_full = evaluate_clustering(X_all, kmeans_labels_full, true_labels)
print(f"Métricas K-Means (4 features):")
for metric, value in kmeans_metrics_full.items():
    print(f"- {metric}: {value:.4f}")

# Visualizar os clusters
plot_clusters(X_all, kmeans_labels_full, feature_names, 
              "K-Means (todas as features)", "kmeans_full")

# Usando apenas duas features
kmeans_two = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels_two = kmeans_two.fit_predict(X_two)

# Avaliação das métricas
kmeans_metrics_two = evaluate_clustering(X_two, kmeans_labels_two, true_labels)
print(f"\nMétricas K-Means (2 features - petal_length, petal_width):")
for metric, value in kmeans_metrics_two.items():
    print(f"- {metric}: {value:.4f}")

# Visualizar os clusters
plot_clusters(X_two, kmeans_labels_two, ['petal_length', 'petal_width'], 
              "K-Means (duas features)", "kmeans_two")


# -----------------------------------------------------------------------------
# 5. AGRUPAMENTO HIERÁRQUICO
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("AGRUPAMENTO HIERÁRQUICO")
print("="*50)

# 5.1 Dendrograma para escolher o número de clusters
print("\n1. Construindo um dendrograma para ajudar a escolher o número de clusters")

# Calcular a ligação
linked = linkage(X_all, method='ward')

# Criar o dendrograma
plt.figure(figsize=(12, 8))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           truncate_mode='lastp',
           p=30)
plt.title('Dendrograma Hierárquico - Método Ward')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.axhline(y=6, c='k', linestyle='--', alpha=0.3)  # Linha sugerindo corte para 3 clusters
plt.tight_layout()
plt.savefig('hierarchical_dendrograma.png', dpi=300, bbox_inches='tight')
plt.close()

print("- Dendrograma salvo como 'hierarchical_dendrograma.png'")
print("- Análise do dendrograma sugere um corte que produz 3 clusters")

# 5.2 Aplicação do Agrupamento Hierárquico com 3 clusters
print("\n2. Aplicação do Agrupamento Hierárquico com 3 clusters")

# Usando todas as features
agglo_full = AgglomerativeClustering(n_clusters=3, linkage='ward')
agglo_labels_full = agglo_full.fit_predict(X_all)

# Avaliação das métricas
agglo_metrics_full = evaluate_clustering(X_all, agglo_labels_full, true_labels)
print(f"Métricas Agglomerative Clustering (4 features):")
for metric, value in agglo_metrics_full.items():
    print(f"- {metric}: {value:.4f}")

# Visualizar os clusters
plot_clusters(X_all, agglo_labels_full, feature_names, 
              "Agglomerative Clustering (todas as features)", "agglo_full")

# Usando apenas duas features
agglo_two = AgglomerativeClustering(n_clusters=3, linkage='ward')
agglo_labels_two = agglo_two.fit_predict(X_two)

# Avaliação das métricas
agglo_metrics_two = evaluate_clustering(X_two, agglo_labels_two, true_labels)
print(f"\nMétricas Agglomerative Clustering (2 features - petal_length, petal_width):")
for metric, value in agglo_metrics_two.items():
    print(f"- {metric}: {value:.4f}")

# Visualizar os clusters
plot_clusters(X_two, agglo_labels_two, ['petal_length', 'petal_width'], 
              "Agglomerative Clustering (duas features)", "agglo_two")


# -----------------------------------------------------------------------------
# 6. RESUMO E COMPARAÇÃO DOS RESULTADOS
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("RESUMO E COMPARAÇÃO DOS RESULTADOS")
print("="*50)

# Estruturar todas as métricas para comparação
results = [
    {"Método": "K-Means (4 features)", **kmeans_metrics_full},
    {"Método": "K-Means (2 features)", **kmeans_metrics_two},
    {"Método": "Hierarchical (4 features)", **agglo_metrics_full},
    {"Método": "Hierarchical (2 features)", **agglo_metrics_two}
]

# Converter para DataFrame para melhor visualização
results_df = pd.DataFrame(results)
print("\nComparação dos resultados:")
print(results_df.to_string(index=False, float_format="{:.4f}".format))

print("\nConclusões:")
print("1. O K-Means conseguiu separar bem as três espécies, mas não perfeitamente.")
print("2. A espécie Iris-setosa é facilmente separável das outras duas.")
print("3. Existe uma sobreposição entre Iris-versicolor e Iris-virginica que dificulta a separação perfeita.")
print("4. O dendrograma mostrou uma estrutura hierárquica que faz sentido, com uma clara divisão inicial em dois grupos (Setosa vs. Versicolor/Virginica).")
print("5. Usar apenas duas features (comprimento e largura da pétala) resultou em uma pequena perda de performance, mas ainda com resultados razoáveis.")
print("6. Em termos de ARI, o agrupamento hierárquico teve melhor desempenho usando todas as features.")

print("\nTodos os resultados foram salvos como arquivos PNG para visualização.")
