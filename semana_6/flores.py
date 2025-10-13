import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris

# Cargar datos
iris = load_iris()
X = iris.data[:, 2:4]  # Solo Longitud y Ancho del pétalo (columnas 2 y 3)
y = (iris.target == 0).astype(np.float32).reshape(-1, 1)  # Setosa=1, No setosa=0

# Convertir datos a tensores tf.float32
train_in = tf.constant(X, dtype=tf.float32)
train_out = tf.constant(y, dtype=tf.float32)

# Añadir una columna de 1s para el sesgo (bias) en la variable de entrada
ones = tf.ones(shape=(train_in.shape[0], 1), dtype=tf.float32)
train_in = tf.concat([train_in, ones], axis=1)  # Ahora dimensión [n_muestras, 3]

# Variable pesos inicializada aleatoriamente [3,1] (2 características + bias)
w = tf.Variable(tf.random.normal([3, 1], seed=12))

# Definición función de modelo (usar sigmoid para clasificación binaria)
def model(x):
    logits = tf.matmul(x, w)
    return tf.sigmoid(logits)

# Función pérdida con entropía cruzada para clasificación binaria
def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

# Optimizador SGD
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Entrenamiento con GradientTape
training_epochs = 1000
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        y_pred = model(train_in)
        loss = loss_fn(y_pred, train_out)
    gradients = tape.gradient(loss, [w])
    optimizer.apply_gradients(zip(gradients, [w]))
    if training_epochs >= 990:
        print('Epoch--', epoch, '--loss--', loss.numpy())

# Para probar el modelo final con un ejemplo
y_pred_final = model(train_in)
print("Predicciones finales (primeras 10):", y_pred_final.numpy()[:10].round())
