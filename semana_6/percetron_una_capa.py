import tensorflow as tf

# Datos de entrada y salida
train_in = tf.constant([
    [1., 1., 1.],
    [1., 0., 1.],
    [0., 1., 1.],
    [0., 0., 1.]
], dtype=tf.float32)

train_out = tf.constant([
    [1.],
    [0.],
    [0.],
    [0.]
], dtype=tf.float32)

# Variable pesos inicializada aleatoriamente
w = tf.Variable(tf.random.normal([3, 1], seed=12))

# Definición función de modelo
def model(x):
    return tf.nn.relu(tf.matmul(x, w))

# Definición función pérdida (error cuadrático medio)
def loss_fn(y_pred, y_true):
    return tf.reduce_sum(tf.square(y_pred - y_true))

# Optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Entrenamiento con GradientTape
training_epochs = 1000
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        y_pred = model(train_in)
        loss = loss_fn(y_pred, train_out)
    gradients = tape.gradient(loss, [w])
    optimizer.apply_gradients(zip(gradients, [w]))
    if training_epochs > 990:
        print('Epoch--', epoch, '--loss--', loss.numpy())
