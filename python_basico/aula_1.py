# -*- coding: utf-8 -*-
"""
Aula 1 de LPAA - Linguagem de Programação Aplicada a Automação - 2020.1

- Generalidades da Linguagem Python
- Comandos Básicos


Professor: Emmanuel Andrade
"""

#Importanto módulos e bibliotecas

import numpy as np
import pandas as pd


#Passagem de argumentos

a = [1,2,3]
b = a.copy()

a.append(4)

b = [1,2]

# Formatos e tipos

a=5

type(a)

b='foo'

type(b)

c= 1.565478

type(c)

d = True

type(d)


#--------

a=4.5

b=2

c = True

print("a é {0}, b é {1}, c é {2} =".format(type(a),type(b),type(c)))

a/b
a**2

a=5

isinstance(a,int)

#-----------

a='foo'

b = a.capitalize()

b

#-----------

#Operadores Binários e comparações


5+3
3-7
5<2
5==5

a=[1,2,3]
b = a
c=list(a)
a is b
a is c

a==c

#----------

#objetos  mutáveis e imutáveis
a_list = ['foo',2,[3,4]]
a_list[2]=[1,2]

# strings e tuplas são imutáveis
a_tuple = (3,55,[4,5])
a_tuple[2]='four'

# procure favorecer a imutabilidade, e evitar efeitos colaterais

# Tipo escalares: Tipos numéricos, bytes strings, e unicodes, booleanos

val = 2

val**3

val_2 = 2e-5
val_2

frase1 = "esta é a frase está com aspas duplas"
frase2 = ' esta é a frase 2 com aspas simples'
frase3 = """esta é uma frase longa
que pode ser expandida em múltiplas
linhas"""

frase1.count('a')
frase1[1]
frase1[0]="E" #strings são imultáveis. Precisa usar o comando replace para substituir a string.

frase1.replace("esta","Esta")

frase1 = "Esta é a frase 1 com aspas duplas"

frase1_lista = list(frase1)
frase1_lista

frase1_lista[10:16] #slicing ou fatiamento

frase4 = 3*frase2  #pesquisar sobre templating ou formatação de string

#casting de tipos

S = '3.14159'

type(S)

fval = float(S)

type(fval)

int(fval)