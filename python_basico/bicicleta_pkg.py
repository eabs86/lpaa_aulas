# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:21:54 2023

@author: apoco
"""

class Bicicleta:
    def __init__(self, cor, modelo, ano, valor, aro=18):
        self.cor = cor
        self.modelo = modelo
        self.ano = ano
        self.valor = valor
        self.aro = aro

    def buzinar(self):
        print("Sai da frente, corno!!")

    def parar(self):
        print("Parando bicicleta...")
        print("Bicicleta Parada!")

    def correr(self):
        print("radamdamdamdammm!!")

    def __str__(self):
        return f"{self.__class__.__name__}: {', '.join([f'{chave}={valor}' for chave,valor in self.__dict__.items()])}"

