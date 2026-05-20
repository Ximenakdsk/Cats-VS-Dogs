import math
import os

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import Sequence


class DirectoryImageSequence(Sequence):
    def __init__(self, filepaths, labels, datagen, target_size, batch_size, shuffle=True):
        self.filepaths = filepaths
        self.labels = np.asarray(labels, dtype="float32")
        self.datagen = datagen
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.filepaths))
        self.rng = np.random.default_rng(42)
        self.class_indices = {"cats": 0, "dogs": 1}
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_filepaths = [self.filepaths[i] for i in batch_indexes]
        batch_labels = self.labels[batch_indexes]

        images = []
        valid_labels = []

        for filepath, label in zip(batch_filepaths, batch_labels):
            try:
                image = load_img(filepath, target_size=self.target_size)
                array = img_to_array(image)
                if self.datagen is not None:
                    array = self.datagen.random_transform(array)
                    array = self.datagen.standardize(array)
                images.append(array)
                valid_labels.append(label)
            except OSError as error:
                print(f"[!] Saltando archivo inválido: {filepath} ({error})")

        if not images:
            raise ValueError("No se pudo cargar ninguna imagen válida en este batch.")

        return np.asarray(images, dtype="float32"), np.asarray(valid_labels, dtype="float32")

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)


def _collect_valid_files(directory):
    filepaths = []
    labels = []
    skipped_files = []
    class_names = sorted(
        entry.name for entry in os.scandir(directory) if entry.is_dir()
    )

    class_indices = {class_name: index for index, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for filename in sorted(os.listdir(class_dir)):
            filepath = os.path.join(class_dir, filename)
            if not os.path.isfile(filepath):
                continue

            try:
                with Image.open(filepath) as image:
                    image.verify()
                filepaths.append(filepath)
                labels.append(class_indices[class_name])
            except OSError as error:
                skipped_files.append((filepath, str(error)))

    return filepaths, labels, class_indices, skipped_files

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

    # 3. Cargamos los archivos válidos y omitimos las imágenes corruptas.
    print("Cargando imágenes de entrenamiento...")
    train_filepaths, train_labels, class_indices, skipped_train = _collect_valid_files(train_dir)
    if skipped_train:
        print(f"[!] Se omitieron {len(skipped_train)} archivos inválidos en entrenamiento.")

    train_generator = DirectoryImageSequence(
        train_filepaths,
        train_labels,
        train_datagen,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
    )

    print("Cargando imágenes de validación...")
    validation_filepaths, validation_labels, _, skipped_validation = _collect_valid_files(validation_dir)
    if skipped_validation:
        print(f"[!] Se omitieron {len(skipped_validation)} archivos inválidos en validación.")

    validation_generator = DirectoryImageSequence(
        validation_filepaths,
        validation_labels,
        validation_datagen,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
    )

    train_generator.class_indices = class_indices
    validation_generator.class_indices = class_indices

    return train_generator, validation_generator

# (Opcional) Código para probar el script independientemente
if __name__ == '__main__':
    # Obtener el directorio actual donde se asume que están las carpetas 'train' y 'validation'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_gen, val_gen = get_data_generators(current_dir)
