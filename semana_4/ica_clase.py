from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt
# Generación de señales aleatoriamente
np.random.seed(0)
n_samples = 200
time = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * time) # Señal 1
s2 = np.sign(np.sin(3 * time)) # Señal 2
s3 = np.random.randn(n_samples) # Señal 3
S = np.c_[s1, s2, s3] # Matriz combinada
A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])
X = np.dot(S, A.T) # Señales combinadas
# Aplicación ICA
ica = FastICA(n_components=3)
independent_components = ica.fit_transform(X)
# Visualización de los componentes individuales
plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.title('Señales Originales')
plt.plot(S)
plt.subplot(4, 1, 2)
plt.title('Señales Combinadas')
plt.plot(X)


plt.subplot(4, 1, 3)
plt.title('Componentes ICA')
plt.plot(independent_components)
plt.subplot(4, 1, 4)
plt.title("Señales Originales (Antes ICA)")
reconstructed_signals = np.dot(independent_components, A) 
plt.plot(reconstructed_signals)
plt.tight_layout()
plt.show()