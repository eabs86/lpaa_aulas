import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do circuito RC
R = 1000  # Resistência em ohms
C = 0.001  # Capacitância em farads

# Frequências de corte
frequencies = [10, 100, 1000,10000]  # Hz

# Vetor de frequências em escala logarítmica
frequencies_log = np.logspace(0, 5, 1000000)  # Varia de 1 Hz a 10 kHz

# Função para calcular a resposta em frequência
def rc_filter(frequency, f0):
    f_corte = f0
    H = 1 / np.sqrt(1 + (frequency/f_corte) ** 2)
    return 20 * np.log10(np.abs(H))

# Plotagem das respostas em frequência
plt.figure(figsize=(10, 6))
for f in frequencies:
    print(f)
    response = rc_filter(frequencies_log,f)
    plt.semilogx(frequencies_log, response, label=f'{f} Hz')

# Configurações do gráfico
plt.xlabel('Frequência (Hz)')
plt.axhline(y=-3, color='red', linestyle='--', label='-3 dB')
plt.xscale('log')
plt.ylabel('Magnitude (dB)')
plt.title('Resposta em Frequência de Filtro Passa-Baixa RC')
plt.grid(True)
plt.legend()
plt.show()
