import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#Cargar los datos de entrada
Estudiantes = pd.read_csv('Estudiantes.csv', index_col=0) #index_col=0 indicamos que las filas tienen un nombre

#print(Estudiantes)

#Estandarizar los datos
#Crear un objeto StandardScalar
scaler = StandardScaler(with_mean=True, with_std=True)

#Ajustar el scalar a los datos y transformar los datos estandarizados
Estudiantes_std = scaler.fit_transform(Estudiantes)

#Imprimir los datos estandarizados
#print(Estudiantes_std)

#Le pedimos que nos muestre la media y la desviación estandar de los datos
# print(np.mean(Estudiantes_std[:,0]))
# print(np.std(Estudiantes_std[:,0]))


#Llevamos a cabo el Análisis de componentes Principales

#Creamos el modelo PCA indicadole que queremos solamente 2 componentes principales
pca = PCA(n_components=2)

#Obtenemos los componentes principales
Estudiantes_pca = pca.fit_transform(Estudiantes_std)

#Imprimir los datos transformados
#print(Estudiantes_pca)

#Varianza explicaa por cada componente
#print(pca.explained_variance_ratio_)

#Varianza explicada entre los 2 componentes principales
#print(np.sum(pca.explained_variance_ratio_))

#Ubicamos a los estudiantes en el plano bidimensional
fig = plt.figure(figsize=(5,5))
plt.rcParams['font.family'] = 'serif'

x_label = 'Componente 1 (' + str(round(pca.explained_variance_ratio_[0]*100,2))+'%)'
y_label = 'Componente 2 (' + str(round(pca.explained_variance_ratio_[1]*100,2))+'%)'

nombres = Estudiantes.index

ax = fig.add_subplot(1,1,1)
ax.set_xlabel(x_label, fontsize = 10)
ax.set_ylabel(y_label, fontsize = 10)
ax.set_title('Componentes principales', fontsize = 15, fontstyle = 'italic')
ax.set_xlim(-3,4)

ax.scatter(x = Estudiantes_pca[:, 0], y = Estudiantes_pca[:, 1], s= 20)

for i, nombre in enumerate(nombres):
    ax.annotate(nombre, (Estudiantes_pca[i, 0]-0.2, Estudiantes_pca[i, 1]+0.05), fontsize = 8)

plt.show()
   