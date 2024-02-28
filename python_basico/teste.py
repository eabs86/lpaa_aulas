import pandas as pd
import matplotlib.pyplot as plt

# Passo 1: Carregar a base de dados
url = "http://worldhappiness.report/ed/2022/"
df = pd.read_csv(url)

# Passo 2: Exibir as 5 primeiras linhas
print("5 primeiras linhas do DataFrame:")
print(df.head())

# Passo 3: Função para análise de dados
def analise_dados(dataframe):
    print("\nAnálise de Dados:")
    num_paises = len(dataframe)
    num_colunas = len(dataframe.columns)
    colunas = dataframe.columns.tolist()
    print(f"Número total de países no dataset: {num_paises}")
    print(f"Número de colunas no dataset: {num_colunas}")
    print(f"Lista de colunas presentes no dataset: {colunas}")
    print("\nContagem de valores ausentes por coluna:")
    print(dataframe.isnull().sum())

analise_dados(df)

# Passo 4: Função para os top 10 países em felicidade
def top_10_paises_felicidade(dataframe):
    top_10 = dataframe.nlargest(10, 'Score')[['Country name', 'Score']]
    print("\nTop 10 países em pontuação de felicidade:")
    print(top_10)

top_10_paises_felicidade(df)

# Passo 5: Função para plotar a pontuação de felicidade ao longo dos anos
def plotar_felicidade_por_ano(dataframe):
    pais = input("\nDigite o nome do país para visualizar sua pontuação de felicidade ao longo dos anos: ")
    pais_data = dataframe[dataframe['Country name'] == pais]
    if not pais_data.empty:
        anos = pais_data['year']
        pontuacao = pais_data['Score']
        plt.plot(anos, pontuacao)
        plt.xlabel('Ano')
        plt.ylabel('Pontuação de Felicidade')
        plt.title(f'Pontuação de Felicidade de {pais} ao longo dos anos')
        plt.show()
    else:
        print(f"País '{pais}' não encontrado no dataset.")

plotar_felicidade_por_ano(df)

# Passo 6: Classe para estatísticas de felicidade
class EstatisticasFelicidade:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def media_felicidade_global(self):
        return self.dataframe['Score'].mean()

    def desvio_padrao_felicidade_global(self):
        return self.dataframe['Score'].std()

    def pais_com_maior_felicidade_por_ano(self):
        return self.dataframe.groupby('year')['Country name', 'Score'].apply(lambda x: x.nlargest(1, 'Score')).reset_index(drop=True)

    def ano_com_maior_media_felicidade(self):
        return self.dataframe.groupby('year')['Score'].mean().idxmax()

estatisticas = EstatisticasFelicidade(df)
print("\nEstatísticas de Felicidade:")
print(f"Média da pontuação de felicidade global: {estatisticas.media_felicidade_global()}")
print(f"Desvio padrão da pontuação de felicidade global: {estatisticas.desvio_padrao_felicidade_global()}")
print(f"País com a maior pontuação de felicidade em cada ano:")
print(estatisticas.pais_com_maior_felicidade_por_ano())
print(f"Ano com a maior pontuação de felicidade média: {estatisticas.ano_com_maior_media_felicidade()}")

# Passo 7: Chamar as funções e métodos
