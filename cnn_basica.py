from tensorflow.keras import layers, models, optimizers

def build_simple_cnn(input_shape=(150, 150, 3)):
    """
    Construye una Red Neuronal Convolucional (CNN) básica desde cero.
    Esta arquitectura es una secuencia de capas convolucionales (para extraer características)
    y capas de agrupación (para reducir el tamaño de las imágenes), terminando en capas
    densas para la clasificación final.
    """
    model = models.Sequential()

    # BLOQUE 1
    # Capa convolucional: busca características (bordes, texturas) usando 32 filtros de 3x3 píxeles
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # Capa de agrupación (Pooling): reduce la imagen a la mitad, quedándose con los valores máximos
    model.add(layers.MaxPooling2D((2, 2)))

    # BLOQUE 2
    # Aumentamos los filtros a 64 para aprender características más complejas
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # BLOQUE 3
    # Aumentamos los filtros a 128
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # BLOQUE 4
    # Un bloque más de 128 filtros para capturar detalles aún más abstractos (ej. orejas, hocicos)
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # FLATTEN
    # Aplana los mapas de características 2D resultantes en un vector 1D
    # para que puedan entrar a la red neuronal tradicional (capas densas)
    model.add(layers.Flatten())

    # DROPOUT
    # "Apaga" aleatoriamente el 50% de las neuronas en cada paso de entrenamiento.
    # Es otra técnica clave para evitar el sobreajuste (overfitting).
    model.add(layers.Dropout(0.5))

    # CAPA DENSA (Oculta)
    # Red neuronal tradicional con 512 neuronas
    model.add(layers.Dense(512, activation='relu'))

    # CAPA DE SALIDA
    # Al ser clasificación binaria (Gato o Perro), solo necesitamos 1 neurona.
    # La activación 'sigmoid' nos da una probabilidad entre 0 y 1 (ej. <0.5 Gato, >0.5 Perro)
    model.add(layers.Dense(1, activation='sigmoid'))

    # COMPILACIÓN DEL MODELO
    # Definimos cómo aprenderá el modelo.
    # - loss: usamos 'binary_crossentropy' porque es un problema de dos clases.
    # - optimizer: 'adam' es muy popular y ajusta la tasa de aprendizaje automáticamente.
    # - metrics: nos interesa ver la precisión ('accuracy').
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

if __name__ == '__main__':
    # Si ejecutamos este script directamente, imprimirá el resumen (arquitectura) del modelo
    modelo_basico = build_simple_cnn()
    print("=== Arquitectura de la CNN Básica ===")
    modelo_basico.summary()
