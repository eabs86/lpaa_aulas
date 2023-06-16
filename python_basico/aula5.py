# -*- coding: utf-8 -*-
"""
LPAA - Linguagem de Programação Aplicada a Automação - 2023.1

- Pandas (Comandos e Generalidades)


Professor: Emmanuel Andrade
"""

""""
Aula sobre limpeza e preparação de dados utilizando pandas

- Preparação de dados ocupa cerca de 80% ou mais tempo na hora de se trabalhar
com dados.


"""

import pandas as pd
import numpy as np


# Tratando dados ausentes

string_data = pd.Series(['a','b',np.nan,'c'])

string_data.isnull()

string_data[0]=None

string_data

string_data.isnull()

#filtrando dados ausentes

from numpy import nan as NA

data = pd.Series([1,NA,3.5,NA,7])

data

data_aux = data.dropna()

data[data.notnull()] #por indexação

#com dataframes

data = pd.DataFrame([[1.,6.5,3],[1, NA, NA],[NA,NA,NA],[NA,6,3]])

data

cleaned = data.dropna()

cleaned

data

data.dropna(how = 'all') #descarta apenas colunas que contenham somenta NAs. Nesse caso descartou a linha

data[4]=NA

data

data.dropna(axis=1, how='all') #nesse caso descartou a coluna axis = 0 ele descarta a linha


#comando thresh

df = pd.DataFrame(np.random.randn(7,3))

df.iloc[:4,1]=NA

df.iloc[:2,2] = NA

df.iloc[0,0] = NA

df

df.dropna()

df.dropna(thresh=3) #exclui  1 determinado numero de observações

#preenchendo dados ausentes

df.fillna(0)

df.fillna({0:0.7,1:0.5,2:0})

_=df.fillna(0,inplace=True)

df

#métodos de interpolação

df = pd.DataFrame(np.random.randn(6,3))

df.iloc[2:,1]=NA

df.iloc[4:,2] = NA

df

df.fillna(method='ffill')


df.fillna(method='ffill',limit=2)

data = pd.Series([1.,NA,3,NA,7])

_=data.fillna(data.mean(),inplace=True)


"""
Transformação de Dados

"""
#removendo duplicatas

data = pd.DataFrame({'k1':['one','two']*3+['two']+['one']+['two'],
                     'k2':[1,1,2,3,3,4,4,1,4]})

data.duplicated(subset='k2') #retorna se a linha é duplicata foi observada ou não em uma linha anterior

data.drop_duplicates()

#esses métodos anteriores consideram todas as colunas

data['v1']=range(9)

data.drop_duplicates(['k1']) #mantém a primeira combinação de valores observados

data.drop_duplicates(['k1','k2'],keep='last')

"""
Transformação de Dados usando uma função ou mapeamento

"""

data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                              'Pastrami', 'corned beef', 'Bacon',
                              'pastrami', 'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data

meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}

lowercased = data['food'].str.lower()
lowercased
data['animal'] = lowercased.map(meat_to_animal)
data

data['food'].map(lambda x: meat_to_animal[x.lower()])

"""
Renomeando índices e eixos

"""


data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])

transform = lambda x: x[:4].upper()
data.index.map(transform)

data.index = data.index.map(transform)
data

data.rename(index=str.title, columns=str.upper)

data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})

data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data

"""
Detectando e filtrando Outliers

"""

data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()

col = data[2]
col[np.abs(col) > 3]

data[(np.abs(data) > 3).any(1)]

data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()

np.sign(data).head()