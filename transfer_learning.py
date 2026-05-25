from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers

def build_vgg16_transfer_learning(input_shape=(150, 150, 3)):
    """
    Construye un modelo de Transfer Learning basado en VGG16 para clasificación binaria.

    Carga VGG16 pre-entrenada con ImageNet sin su cabeza original (include_top=False)
    y congela todas sus capas para preservar las características aprendidas. Sobre la
    base congelada agrega: Flatten, Dense(256, ReLU), Dropout(0.5) y salida sigmoid.
    Se compila con RMSprop(lr=2e-5) para ajuste fino sin sobrescribir los pesos base.

    Retorna: modelo Sequential compilado listo para entrenar.
    """
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )
    # Congela la base para preservar los pesos pre-entrenados
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # lr=2e-5 para ajuste fino sin sobrescribir los pesos pre-entrenados
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=2e-5),
        metrics=['accuracy'],
    )

    return model

if __name__ == '__main__':
    modelo_tl = build_vgg16_transfer_learning()
    print("=== Arquitectura Transfer Learning (VGG16) ===")
    modelo_tl.summary()
