
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
data = np.random.randn(10000) * 3 + 1
plt.hist(data)
plt.xlabel("Wartosci")
plt.ylabel("Czestotliwodc")
plt.title("Histogram danych z rozkladu normalnego (μ=1, σ=3)")
plt.show()


