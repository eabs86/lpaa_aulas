# -*- coding: utf-8 -*-
"""
LPAA - Linguagem de Programação Aplicada a Automação - 2023.1

- Numpy (Comandos e Generalidades)


Professor: Emmanuel Andrade
"""

import numpy as np

# Criando ndarrays unidimensional

data1 = [1,2,3,4.5,5,6]

array = np.array(data1)

array.dtype
array.ndim

type(data1)

type(array)

# Criando ndarray bidimensional




data2 = [[1,2,3,4],[5,6,7,8]]

array2d = np.array(data2)

array2d.ndim

array2d.shape
array2d_reshaped = array2d.reshape(4,2)

array.dtype

array2d.dtype

np.zeros((10,2))

np.zeros((2,3))

zeros3D = np.zeros((2,3,3))

np.ones((4,4))

np.identity(5)

np.empty(5) # cuidado!!!

array_arange = np.arange(15)

#Aritmética com Arrays

arrayx10 = array*10

array2d-array2d

array2d - 5

array2d*array2d

array2d**2

data3 = [[0.2,5,3,2],[1,4,0.8,-2]]

array3 = np.array(data3)

teste = array3<array2d

#Indexação e Fatiamento

arr = np.arange(10)

arr[5]

arr[5:8]

arr_slice = arr[5:8]

arr_slice_copy=arr_slice.copy()

arr_slice_copy[1]=12345

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])


arr2d[2]

arr2d[0][2]

arr2d[0,2]

arr2d[1:,1:]

arr2d[:2,1:]

#Indexação Booleana


nomes = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])

data=np.random.rand(7,4)

data

nomes=='Bob'

indexador = nomes!='Bob'

temp=data[indexador]


array = np.array([1, 2, 3, 4, 5])

mean_value = np.mean(array)
max_value = np.max(array)
var_value = np.var(array)
std_value = np.std(array)
min_value = np.min(array)
sum_value = np.sum(array)
sqrt_value = np.sqrt(array)
dot_product = np.dot(array, array)

print(mean_value)
print(max_value)
print(min_value)
print(sum_value)
print(dot_product)

samples = np.random.randint(4,10,size=(4,4))

np.random.seed(10)
