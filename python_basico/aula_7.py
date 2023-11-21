""" 
LPAA - Linguagem de Programação Aplicada a Automação - 2023.1

- Funções e POO em python


Professor: Emmanuel Andrade


"""

"""

Trabalhando com Funções

"""

# Função que exibe uma mensagem de boas-vindas padrão
def exibir_mensagem():
    print("Seja bem vindo!")

# Função que exibe uma mensagem de boas-vindas com um nome específico
def exibir_mensagem_2(nome):
    print(f"Seja bem vindo {nome}!")

# Função que exibe uma mensagem de boas-vindas com um nome (opcionalmente "Anônimo" se não for fornecido)
def exibir_mensagem_3(nome="Anônimo"):
    print(f"Seja bem vindo {nome}!")

# Chamada das funções de exibição de mensagens
exibir_mensagem()
exibir_mensagem_2("Fulano")
exibir_mensagem_3()
exibir_mensagem_3(nome="Ciclano")

# Função que calcula a soma de uma lista de números
def calcular_total(numeros):
    return sum(numeros)

# Função que retorna o antecessor e o sucessor de um número
def retorna_antecessor_e_sucessor(numero):
    antecessor = numero - 1
    sucessor = numero + 1
    return antecessor, sucessor

# Chamadas das funções com retorno
calcular_total([10, 20, 34])  # 64
retorna_antecessor_e_sucessor(10)  # (9, 11)

# Função que exibe um poema com informações de data, texto e metadados
def exibir_poema(data_extenso, *args, **kwargs):
    texto = "\n".join(args)
    meta_dados = "\n".join(
        [f"{chave.title()}: {valor}" for chave, valor in kwargs.items()])
    mensagem = f"{data_extenso}\n\n{texto}\n\n{meta_dados}"
    print(mensagem)

# Chamada da função com argumentos variáveis
exibir_poema("Zen of Python", "Beautiful is better than ugly.",
             autor="Tim Peters", ano=1999)

# Função com argumentos posicionais apenas
def criar_carro_po(modelo, ano, placa, /, marca, motor, combustivel):
    print(modelo, ano, placa, marca, motor, combustivel)

# Chamadas da função com argumentos posicionais e de palavra-chave
criar_carro_po("Palio", 1999, "ABC-1234", marca="Fiat",
            motor="1.0", combustivel="Gasolina")  # válido
# A seguinte chamada é inválida devido ao uso incorreto de argumentos de palavra-chave
criar_carro_po(modelo="Palio", ano=1999, placa="ABC-1234", marca="Fiat",
            motor="1.0", combustivel="Gasolina")  # inválido

# Função com argumentos de palavra-chave apenas
def criar_carro_ko(*, modelo, ano, placa, marca, motor, combustivel):
    print(modelo, ano, placa, marca, motor, combustivel)

# Chamadas da função com argumentos de palavra-chave
criar_carro_ko(modelo="Palio", ano=1999, placa="ABC-1234",
            marca="Fiat", motor="1.0", combustivel="Gasolina")  # válido
# A seguinte chamada é inválida devido ao uso incorreto de argumentos posicionais
criar_carro_ko("Palio", 1999, "ABC-1234", marca="Fiat",
            motor="1.0", combustivel="Gasolina")  # inválido

# Função com argumentos posicionais e de palavra-chave
def criar_carro_complete(modelo, ano, placa, /, *, marca, motor, combustivel):
    print(modelo, ano, placa, marca, motor, combustivel)

# Chamadas da função com argumentos posicionais e de palavra-chave
criar_carro_complete("Palio", 1999, "ABC-1234", marca="Fiat",
            motor="1.0", combustivel="Gasolina")  # válido
# A seguinte chamada é inválida devido ao uso incorreto de argumentos posicionais
criar_carro_complete(modelo="Palio", ano=1999, placa="ABC-1234", marca="Fiat",
            motor="1.0", combustivel="Gasolina")  # inválido

# Classe que representa uma calculadora simples
class Calculadora:
    def __init__(self):
        self.valor = 0

    def somar(self, num):
        self.valor += num

    def subtrair(self, num):
        self.valor -= num

    def obter_resultado(self):
        return self.valor

# Criar uma instância da classe Calculadora
calculadora = Calculadora()
calculadora.somar(5)
calculadora.subtrair(3)
resultado = calculadora.obter_resultado()
print("O resultado da calculadora é:", resultado)

# Classe que representa uma bicicleta
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

# from bicicleta_pkg import Bicicleta

from bicicleta_pkg import *


b1 = Bicicleta("vermelha", "caloi", 2022, 600)
b1.buzinar()
b1.correr()
b1.parar()
print(b1)
print(b1.cor, b1.modelo, b1.ano, b1.valor)
