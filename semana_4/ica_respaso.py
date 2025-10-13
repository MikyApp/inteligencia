import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Crear señales originales independientes (fuentes)
np.random.seed(0)
S = np.array([np.sin(2 * np.pi * 0.02 * np.arange(200)),
              np.sign(np.sin(3 * np.pi * 0.05 * np.arange(200))),
              np.random.randn(200)])

# Mezcla de las señales originales (matriz de mezcla)
A = np.array([[1, 1, 0.5], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
X = A.dot(S)

# Aplicar ICA para recuperar las señales originales
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X.T).T  # Reconstuir señales separadas

# Graficar señales originales, mezcladas y separadas
plt.figure(figsize=(10, 7))
models = [S, X, S_]
titles = ['Señales Originales', 'Señales Mezcladas', 'Señales Recuperadas por ICA']
for i, (model, title) in enumerate(zip(models, titles), 1):
    plt.subplot(3, 1, i)
    plt.title(title)
    for sig in model:
        plt.plot(sig)
plt.tight_layout()
plt.show()
