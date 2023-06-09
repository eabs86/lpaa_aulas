# -*- coding: utf-8 -*-
"""
LPAA - Linguagem de Programação Aplicada a Automação - 2023.1

- Pandas (Comandos e Generalidades)


Professor: Emmanuel Andrade

- Biblioteca projetada para trabalhar com dados tabulares e heterogêneos.

- Numpy é bom para trabalhar com dados numéricos homogêneos em arrays

"""
import numpy as np
import pandas as pd

#Principais estrutura de dados: Series e DataFrame

obj = pd.Series([4,7,-5,3])

obj.values #valores

obj.index #indexadores

obj2 = pd.Series([4,7,-5,3], index=['d','b','a','c'])

obj2['a']

obj3 = obj2.copy() #cuidado na atribuição

obj2['a']=-4



#indexação utilizando uma lista de índices
obj2[['c','a','b']]

obj2[obj2>0]

obj2*2

np.exp(obj2)

'b' in obj2

'e' in obj2


# Dados em um dicionário

sdata = {'Ohio':35000, 'Texas': 71000, 'Oregon':21654, 'Utah':50000}

obj3 = pd.Series(sdata)

states = ['California','Ohio','Oregon','Texas']

obj4=pd.Series(sdata,index=states) #construiu um objeto sem Utah, pois não consta no dicionário

pd.isnull(obj4) #informa que o primeiro item é vazio

pd.notnull(obj4) #informa que o primeiro item não é vazio

obj4.isnull()

obj5 = obj3 + obj4

obj4.name = 'population'

obj4

obj.index =['Bob','Steve','Jeff','Ryan']

obj

"""
Dataframe: Representa uma tabela de dados retangular que contém uma coleção 
ordenada de colunas, em que cada uma pode ter um tipo de valor de diferentes
tipos.

É como um dicionário de séries, onde todos compartilham um mesmo índice.

Fisicamente ele é bidimensional, mas pode ser utilizado para dimensões maiores
através de indexação hierárquica 

"""

data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada','Nevada'],
        'year':[2000,2001,2002,2001,2002,2003],
        'pop':[1.5,1.7,3.6,2.4,2.9,3.2]
        }

frame = pd.DataFrame(data)

frame.head() #exibe somente as 5 primeiras linhas
frame.tail()

pd.DataFrame(data,columns=['year', 'state','pop']) #coloca as colunas numa sequencia específica



frame2 =pd.DataFrame(data,columns=['year','state','pop','debt'],
                     index = ['one','two','three','four','five','six'])

frame2['state']

frame2.state

frame2.loc[['three']]

frame2['debt']=np.arange(6.) #cuidado. Quando for atribuir, precisa ser do mesmo tamanho

val = pd.Series([-1.2,-1.5,-1.7],index=['two','four','five'])

frame2['debt']=val

frame2['eastern']=frame2.state == 'Ohio'

del frame2['eastern'] #qualquer modificação in place em Series se refletirá em DataFrame

#Dicionários aninhados
#Chaves mais externas viram colunas, e as mais internas viram índice das linhas
pop = {
       'Ohio':{2000:1.5,2001:1.7,2002:3.6},
       'Nevada':{2001:2.4,2002:2.9}}

frame3=pd.DataFrame(pop)

frame3.T #transposição

a=frame3.values

frame2.values


#Objetos Index

obj = pd.Series(range(3),index=['a','b','c'])

index = obj.index

index[1]='d' #objetos Index são imutaveis e podem conter rótulos duplicados


"""

Funcionalidades Essenciais
-Mecanismos fundamentais na interação com os dados contidos numa série ou um 
DataFrame.

"""

obj = pd.Series([4.5,7.2,-5.3,3.6],index = ['d','b','a','c'])

obj

obj2 = obj.reindex(['a','b','c','d','e'])

obj2

obj3 = pd.Series(['blue','purple','yellow'],index = [0,2,4])

obj3

obj4=obj3.reindex(range(6),method='ffill') #foward fill por interpolação



frame=pd.DataFrame(np.arange(9).reshape((3,3)),
                   index=['a','c','d'],
                   columns=['Ohio','Texas','California'])

frame2=frame.reindex(['a','b','c','d']) #reindexação

states = ['Texas','Utah','California']

frame.reindex(columns = states)

#Descarte de entradas de eixos

obj = pd.Series(np.arange(5.),index=['a','b','c','d','e'])

obj

new_obj = obj.drop('c')

new_obj

data = pd.DataFrame(np.arange(16).reshape((4,4)),
                    columns=['one','two','three','four'],
                     index = ['Ohio','Colorado','Utah','New York'])

data.drop(['Colorado','Ohio'])

data.drop('Ohio',axis=0)


data.drop(['two','four'],axis='columns')

obj.drop('c',inplace=True)
data.drop(['Colorado','Ohio'],inplace = True)

obj


# Seleção com Iloc
# Permitem selecionar um subconjunto de linhas e colunas de um DataFrame

data= pd.DataFrame(np.arange(16).reshape((4,4)),
                   index=['Ohio', 'Colorado','Utah','New York'],
                   columns = ['one','two','three','four'])


data.loc['Colorado',['two','three']]

data.iloc[2,[3,0,1]]

data.iloc[2]

data.iloc[[1,2],[3,0,1]]

data.loc['Ohio':,'four']

# Aritmética e alinhamento de dados: é feito pelo indexador

s1 = pd.Series([7.3,-2.5,3.4,1.5],index=['a','c','d','e'])


s2 = pd.Series([-2.1,3.6,-1.5,4,3.1],index=['a','c','e','f','g'])


s3 = s1+s2


# com dataframes

df1 = pd.DataFrame(np.arange(9.0).reshape((3,3)),columns=list('bcd'),
                   index = ['Ohio','Texas','Colorado'])

df2 = pd.DataFrame(np.arange(12.0).reshape((4,3)),columns=list('bde'),
                   index = ['Utah','Ohio','Texas','Oregon'])

df3 = df1 + df2
