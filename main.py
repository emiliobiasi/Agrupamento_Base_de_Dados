from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import pandas as pd
import plotly.express as px


def kmeans_bisecting(base_de_dados):
    print("\nProcessando método de agrupamento particional K-Means...")
    elbow(base_de_dados)
    # Definindo o número de clusters desejado
    num_clusters = int(input("\nDigite a quantidade de Clusters com base no Método Elbow: \n"))

    kmeans = BisectingKMeans(n_clusters=num_clusters, random_state=0).fit(base_de_dados)

    # Obtendo os rótulos dos clusters
    rotulos = kmeans.labels_

    # Plotando o gráfico de dispersão dos dados com cores representando os clusters
    plt.scatter(base_de_dados[:, 0], base_de_dados[:, 1], c=rotulos)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agrupamento Bi-secting K-Means')
    plt.show()

    print("\n--------BASE DE DADOS--------")
    print(base_de_dados)
    print("\n-----------ROTULOS----------")
    print(rotulos)


def linkage_hierarquico(base_de_dados):
    print("\nProcessando método de agrupamento hierárquico Linkage...")
    # Aplicando o algoritmo de agrupamento hierárquico com linkage

    matriz_linkage = linkage(base_de_dados, 'complete')
    fig = plt.figure(figsize=(25, 10))
    dendogram = dendrogram(matriz_linkage)
    plt.show()

    matriz_linkage = linkage(base_de_dados, 'single')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(matriz_linkage)
    plt.show()

    matriz_linkage = linkage(base_de_dados, 'ward')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(matriz_linkage)
    plt.show()


def elbow(dados):
    elbow_list = []
    # Testando diferentes números de clusters de 1 a 10
    for num_clusters in range(1, 11):
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
        print("1 - K-Means - bi-secting (particionado)")
        print("2 - Linkage (hierarquico)")
        r = input("Qual método você deseja? ")
    return r


def processar_base_iris():
    print("Processando a base de dados Iris...")
    df = px.data.iris()
    features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]
    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color="species"
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()
    iris = load_iris()
    dados = iris.data
    r = menu_metodo_agrupamento()
    if r == "1":
        kmeans_bisecting(dados)
    elif r == "2":
        linkage_hierarquico(dados)


def processar_base_wine():
    print("Processando a base de dados 2...")
    base2 = load_wine()
    dados = base2.data
    r = menu_metodo_agrupamento()
    if r == "1":
        kmeans_bisecting(dados)
    elif r == "2":
        linkage_hierarquico(dados)


resposta = 0
while resposta not in ["1", "2"]:
    print("\n------ ESCOLHA A BASE DE DADOS ------")
    print("1 - Base Iris")
    print("2 - Base Wine")
    resposta = input("Qual base de dados você deseja? ")

if resposta == "1":
    processar_base_iris()
elif resposta == "2":
    processar_base_wine()
