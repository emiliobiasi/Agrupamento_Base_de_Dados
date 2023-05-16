from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans

# Carregando a base de dados Iris
iris = load_iris()
dados = iris.data

print(dados)
# Definindo o número de clusters para o K-means
num_clusters_kmeans = 3

# Criando o objeto K-means
kmeans = KMeans(n_clusters=num_clusters_kmeans)

# Aplicando o K-means aos dados
kmeans.fit(dados)

# Obtendo os rótulos dos clusters
rotulos_kmeans = kmeans.labels_

# Definindo o número de clusters para o algoritmo hierárquico
num_clusters_hierarquico = 3

# Criando o objeto de agrupamento hierárquico
hierarquico = AgglomerativeClustering(n_clusters=num_clusters_hierarquico)

# Aplicando o agrupamento hierárquico aos dados
rotulos_hierarquico = hierarquico.fit_predict(dados)

# Imprimindo os resultados do K-means
print("Rótulos dos clusters (K-means):")
print(rotulos_kmeans)

# Imprimindo os resultados do agrupamento hierárquico
print("\nRótulos dos clusters (Agrupamento hierárquico):")
print(rotulos_hierarquico)