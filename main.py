from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def kmeans_hierarquico(base_de_dados):
    print("Processando método de agrupamento hierárquico...")
    elbow(base_de_dados)


def kmeans_particional(base_de_dados):
    print("Processando método de agrupamento particional...")
    elbow(base_de_dados)


def elbow(dados):
    elbow_list = []
    # Testando diferentes números de clusters de 1 a 10
    for num_clusters in range(1, 11):
        # Criando o objeto K-means
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(dados)
        # Obtendo a soma dos quadrados das distâncias intra-cluster e adicionando na elbow_list
        elbow_list.append(kmeans.inertia_)

    # Plotando o gráfico do elbow para melhor visualização da quantidade de clusters ideal.
    plt.plot(range(1, 11), elbow_list)
    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS (soma dos quadrados das distancias)')
    plt.title('Método do Elbow')
    plt.show()


def menu_metodo_agrupamento():
    r = 0
    while r not in ["1", "2"]:
        print("\n------ ESCOLHA O MÉTODO ------")
        print("1 - Hierárquico")
        print("2 - Partiocional")
        r = input("Qual método você deseja? ")
    return r


def processar_base_iris():
    print("Processando a base de dados Iris...")
    iris = load_iris()
    dados = iris.data
    r = menu_metodo_agrupamento()
    if r == "1":
        kmeans_hierarquico(dados)
    elif r == "2":
        kmeans_particional(dados)


def processar_base_2():
    print("Processando a base de dados 2...")
    base2 = load_iris()
    dados = base2.data
    r = menu_metodo_agrupamento()
    if r == "1":
        kmeans_hierarquico(dados)
    elif r == "2":
        kmeans_particional(dados)


resposta = 0
while resposta not in ["1", "2"]:
    print("\n------ ESCOLHA A BASE DE DADOS ------")
    print("1 - Base Iris")
    print("2 - Base 2")
    resposta = input("Qual base de dados você deseja? ")

if resposta == "1":
    processar_base_iris()
elif resposta == "2":
    processar_base_2()

'''
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

# Plotando os dados com os rótulos do K-means
print(rotulos_kmeans)
plt.scatter(dados[:, 0], dados[:, 1], c=rotulos_kmeans)
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('K-means - Rótulos dos Clusters')
plt.show()

# Imprimindo os resultados do agrupamento hierárquico
print("\nRótulos dos clusters (Agrupamento hierárquico):")

# Plotando os dados com os rótulos do agrupamento hierárquico
print(rotulos_hierarquico)
plt.scatter(dados[:, 0], dados[:, 1], c=rotulos_hierarquico)
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Agrupamento Hierárquico - Rótulos dos Clusters')
plt.show()
'''
