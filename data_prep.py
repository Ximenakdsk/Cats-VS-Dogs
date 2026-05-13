import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(base_dir, target_size=(150, 150), batch_size=32):
    """
    Esta función prepara y devuelve los generadores de datos para el entrenamiento
    y la validación. Utiliza ImageDataGenerator para aplicar "Data Augmentation"
    a las imágenes de entrenamiento, lo cual ayuda a prevenir el sobreajuste (overfitting).
    """
    
    # Rutas a las carpetas de entrenamiento y validación
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # 1. Data Augmentation para el conjunto de entrenamiento
    # Agregamos transformaciones aleatorias a las imágenes para que el modelo aprenda 
    # a reconocer patrones sin importar la orientación o posición del animal.
    train_datagen = ImageDataGenerator(
        rescale=1./255,             # Normalizamos los valores de los píxeles para que estén entre 0 y 1
        rotation_range=40,          # Rota las imágenes aleatoriamente hasta 40 grados
        width_shift_range=0.2,      # Desplaza la imagen horizontalmente un 20%
        height_shift_range=0.2,     # Desplaza la imagen verticalmente un 20%
        shear_range=0.2,            # Aplica transformaciones de cizalladura (shear)
        zoom_range=0.2,             # Hace un acercamiento o alejamiento aleatorio
        horizontal_flip=True,       # Voltea las imágenes horizontalmente
        fill_mode='nearest'         # Rellena los píxeles vacíos tras las transformaciones
    )

    # 2. Generador para validación (¡SIN Data Augmentation!)
    # Los datos de validación NO deben ser modificados (salvo la normalización) 
    # para evaluar el modelo con datos "puros".
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # 3. Flujo de datos (Flow from directory)
    # Toma las imágenes de las carpetas, las redimensiona al 'target_size' y 
    # las agrupa en lotes (batches).
    print("Cargando imágenes de entrenamiento...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,    # Redimensiona todas las imágenes al tamaño indicado (ej. 150x150)
        batch_size=batch_size,      # Número de imágenes por lote
        class_mode='binary'         # Clasificación binaria (Perro o Gato -> 0 o 1)
    )

    print("Cargando imágenes de validación...")
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator

# (Opcional) Código para probar el script independientemente
if __name__ == '__main__':
    # Obtener el directorio actual donde se asume que están las carpetas 'train' y 'validation'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_gen, val_gen = get_data_generators(current_dir)
