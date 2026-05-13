from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers

def build_vgg16_transfer_learning(input_shape=(150, 150, 3)):
    """
    Construye un modelo utilizando Transfer Learning basado en VGG16.
    Aprovecha una red pre-entrenada con millones de imágenes (ImageNet)
    y la adapta para nuestro problema específico (Perros vs Gatos).
    """
    
    # 1. Cargar el modelo base pre-entrenado (VGG16)
    # weights='imagenet': Usamos los pesos aprendidos del dataset ImageNet (muy efectivo).
    # include_top=False: Quitamos la última parte de la red original (la que clasifica en 1000 categorías)
    # porque nosotros queremos poner nuestra propia capa final para 2 categorías.
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 2. Congelar el modelo base
    # Muy importante: "Congelamos" estas capas para que sus pesos no se modifiquen
    # durante el entrenamiento inicial. Solo queremos entrenar la parte nueva que agregaremos.
    base_model.trainable = False

    # 3. Construir nuestro modelo uniendo la base VGG16 con nuestras capas personalizadas
    model = models.Sequential()
    
    # Agregamos la "base de conocimiento" de VGG16
    model.add(base_model)
    
    # Aplanamos la salida de VGG16 (igual que hicimos en la CNN básica)
    model.add(layers.Flatten())
    
    # Agregamos una capa densa para que aprenda a interpretar las características extraídas por VGG16
    model.add(layers.Dense(256, activation='relu'))
    
    # Dropout para evitar el sobreajuste en esta nueva parte que estamos entrenando
    model.add(layers.Dropout(0.5))
    
    # Capa final para nuestra clasificación binaria
    model.add(layers.Dense(1, activation='sigmoid'))

    # 4. Compilar el modelo
    # Utilizamos un "learning rate" (tasa de aprendizaje) bajo en Transfer Learning 
    # (ej. RMSprop con lr=2e-5) para hacer ajustes finos sin destruir los pesos pre-entrenados.
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=2e-5),
        metrics=['accuracy']
    )

    return model

if __name__ == '__main__':
    # Mostrar la arquitectura
    modelo_tl = build_vgg16_transfer_learning()
    print("=== Arquitectura Transfer Learning (VGG16) ===")
    modelo_tl.summary()
