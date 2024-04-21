import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt('wine.data', delimiter=',')


for i in range(14):
    var = data[:, i]
    print(f"Zmienna {i+1}:")
    print(f"  Wartość minimalna: {np.min(var)}")
    print(f"  Wartość maksymalna: {np.max(var)}")
    print(f"  Średnia: {np.mean(var)}")
    print(f"  Mediana: {np.median(var)}")
    print(f"  Odchylenie standardowe: {np.std(var)}")


odmiana_1 = data[data[:, 0] == 1]
srednie_odmiana_1 = np.mean(odmiana_1[:, 1:], axis=0)
print("Średnie wartości dla win odmiany 1:")
print(srednie_odmiana_1)


plt.hist(data[:, 5], bins=30, alpha=0.6, color='b')
plt.xlabel('Magnesium')
plt.ylabel('Częstość występowania')
plt.title('Histogram zmiennej Magnesium')
plt.grid(True)
plt.show()


for odmiana in range(1, 4):
    odmiana_data = data[data[:, 0] == odmiana]
    plt.hist(odmiana_data[:, 5], bins=30, alpha=0.6, label=f'Odmiana {odmiana}')
plt.xlabel('Magnesium')
plt.ylabel('Częstość występowania')
plt.title('Histogram zmiennej Magnesium dla każdej odmiany win')
plt.legend()
plt.grid(True)
plt.show()


liczebnosc = np.bincount(data[:, 0].astype(int))
plt.bar(range(1, 4), liczebnosc[1:])
plt.xlabel('Odmiana wina')
plt.ylabel('Liczebność')
plt.title('Liczebność win każdej z 3 odmian')
plt.grid(True)
plt.show()


plt.scatter(data[:, 1], data[:, 2], alpha=0.6)
plt.xlabel('Alcohol')
plt.ylabel('Malic acid')
plt.title('Wykres rozrzutu Alcohol vs Malic acid')
plt.grid(True)
plt.show()


colors = ['r', 'g', 'b']
for odmiana in range(1, 4):
    odmiana_data = data[data[:, 0] == odmiana]
    plt.scatter(odmiana_data[:, 1], odmiana_data[:, 2], alpha=0.6, label=f'Odmiana {odmiana}', color=colors[odmiana-1])
plt.xlabel('Alcohol')
plt.ylabel('Malic acid')
plt.title('Wykres rozrzutu Alcohol vs Malic acid z oznaczeniem odmiany wina')
plt.legend()
plt.grid(True)
plt.show()
