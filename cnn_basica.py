from tensorflow.keras import layers, models, optimizers

def build_simple_cnn(input_shape=(150, 150, 3)):
    """
    Construye y compila una CNN desde cero para clasificación binaria (gatos vs. perros).

    Arquitectura: 4 bloques Conv2D(ReLU) + MaxPooling2D con filtros crecientes
    (32 → 64 → 128 → 128), seguidos de Flatten, Dropout(0.5), Dense(512) y
    una salida sigmoid de 1 neurona. Se compila con Adam y binary_crossentropy.

    Retorna: modelo Sequential compilado listo para entrenar.
    """
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    return model

if __name__ == '__main__':
    modelo_basico = build_simple_cnn()
    print("=== Arquitectura de la CNN Básica ===")
    modelo_basico.summary()
