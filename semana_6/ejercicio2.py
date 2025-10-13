import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración de generadores de datos para entrenamiento y validación
train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_data_gen.flow_from_directory(
    'ruta_a_directorio_de_imagenes/train',  # Cambiar a la ruta real
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_data_gen.flow_from_directory(
    'ruta_a_directorio_de_imagenes/train',  # Misma ruta que entrenamiento
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# Modelo CNN para clasificación de emociones
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emociones
])

# Compilación del modelo
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Mostrar resumen del modelo
model.summary()

# Entrenar el modelo
epochs = 30
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Guardar modelo entrenado para uso posterior
model.save('emotion_classifier_model.h5')
