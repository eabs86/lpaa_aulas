# -*- coding: utf-8 -*-
"""
LPAA - Linguagem de Programação Aplicada a Automação - 2023.1

- Pandas (Comandos e Generalidades)


Professor: Emmanuel Andrade
"""

""""
Aula sobre Matplotlib

- Como apresentar os dados graficamente.


"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = np.arange(30)

data

#plotagem padrão. Há diversos parâmetros que podem ser configurados
#É necessário ler a documentaçãos sobre o matplotlib

plt.plot(data)


#Figuras e Subplotagens

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

# plt.plot(np.random.randn(50).cumsum(),'b--s')
# plt.plot(np.random.randn(30).cumsum(),color="red", linestyle="-.",marker="o")

_=ax4.hist(np.random.rand(100),bins=50,color='r',alpha=1.0)
ax2.scatter(np.arange(30),np.arange(30)+3*np.random.randn(30),color='cyan', alpha=0.8)

#criação de figura com grade de subplotagens

fig,axes = plt.subplots(2,3) #2 linhas por 3 colunas

#uso de axes[i,j]

#ajustando espaçamento em torno das subplotagens

plt.subplots_adjust(left =None, bottom = None, right = None, top = None,
                wspace = None, hspace = None)


fig,axes = plt.subplots(2,2, sharex= False, sharey=True)

for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500),bins=50,color='k',alpha=0.5)

plt.subplots_adjust(wspace=0,hspace=0)

#cores e marcadores
'''
ax.plot(x,y, 'g--')

ou

ax.plot(x,y,linestyle = '--', color= 'g')

'''

plt.plot(np.random.randn(30).cumsum(),'r*--')

plt.plot(np.random.randn(30).cumsum(),color = 'orange',
         linestyle='dotted',marker='o')

data = np.random.randn(30).cumsum()


#formas de interpolação

plt.plot(data, 'k--', label='Dados Aleatórios')

plt.plot(data,'b-',drawstyle='steps-post',
         label='steps-post')

plt.legend(loc='best')

'''
Ticks, rótulos e legendas

Uso do xlim, xticks, xticklabels

'''

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.plot(np.random.randn(1000).cumsum())

ticks = ax.set_xticks([0,250,500,750,1000])

labels = ax.set_xticklabels(['one','two','three','four','five'],
                            rotation=30,fontsize='large')


ax.set_title('Meu primeiro Gráfico no Matplotlib')

ax.set_xlabel('Estágios')

ax.set_ylabel('Valores')

###

props = {
        'title':'Meu Primeiro Gráfico',
        'xlabel': 'Eixo x',
        'ylabel': 'Eixo Y'
        
        }

ax.set(**props)

#acrescentando legendas

fig = plt.figure()

ax=fig.add_subplot(1,1,1)

ax.plot(np.random.randn(1000).cumsum(),'k',label='one')

ax.plot(np.random.randn(1000).cumsum(),'r--',label='two')

ax.plot(np.random.randn(1000).cumsum(),'b.',label='three')

ax.legend(loc='lower left')

'''

Pandas 

'''

#----

s = pd.Series(np.random.randn(10).cumsum(),
              index=np.arange(0,100,10))

s.plot()

# no modo DataFrame é plotada cada uma de suuas colunas como uma linha diferente
# na mesma subplotagem, criando legendas automaticamente.
# Há uma série de opções que permitem flexibilidade na hora de plotar

df = pd.DataFrame(np.random.randn(10,4).cumsum(0),
                  columns=['A','B','C','D'],
                  index = np.arange(0,100,10))
df.plot()

plt.plot(df)

#plotagem de barras

fig, axes = plt.subplots (2,1)

data = pd.Series(np.random.randn(16),
                 index = list('abcdefghijklmnop'))

data.plot.bar(ax=axes[0],color='k', alpha=0.7)

data.plot.barh(ax=axes[1],color='r',alpha=0.3)


np.random.seed(12348)

df = pd.DataFrame(np.random.rand(6, 4),
                  index=['one', 'two', 'three', 'four', 'five', 'six'],
                  columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df
df.plot.bar()

plt.figure()

df.plot.barh(stacked=True, alpha=0.5)

plt.close('all')


