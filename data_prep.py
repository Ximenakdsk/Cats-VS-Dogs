import math
import os

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import Sequence


class DirectoryImageSequence(Sequence):
    """
    Generador de imágenes compatible con Keras que carga los datos por lotes desde disco.

    Evita cargar todo el dataset en memoria a la vez. Al final de cada época reordena
    los índices con una semilla fija (42) para garantizar reproducibilidad. Las imágenes
    corruptas se omiten en tiempo de ejecución sin interrumpir el entrenamiento.
    """

    def __init__(self, filepaths, labels, datagen, target_size, batch_size, shuffle=True):
        """Inicializa el generador y ejecuta el primer shuffle de índices."""
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
        """Devuelve el número de batches por época (redondeo hacia arriba)."""
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, index):
        """
        Carga y devuelve el batch en la posición `index`.

        Aplica las transformaciones del datagen (augmentation o solo rescale) a cada imagen.
        Si un archivo no puede abrirse, se omite y se imprime una advertencia.
        Lanza ValueError si ninguna imagen del batch es válida.
        """
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
        """Reordena aleatoriamente los índices al terminar cada época."""
        if self.shuffle:
            self.rng.shuffle(self.indexes)


def _collect_valid_files(directory):
    """
    Recorre los subdirectorios de `directory` y recopila las rutas de imágenes válidas.

    Cada subdirectorio representa una clase (ej. cats/, dogs/). Las clases se ordenan
    alfabéticamente y se mapean a índices enteros. Cada archivo se abre con PIL para
    verificar su integridad; los archivos corruptos se registran en `skipped_files`
    sin detener la ejecución.

    Retorna: (filepaths, labels, class_indices, skipped_files)
    """
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
    Construye y retorna los generadores de entrenamiento y validación.

    El generador de entrenamiento aplica Data Augmentation (rotación hasta 40°,
    desplazamientos, shear, zoom y flip horizontal) para reducir el sobreajuste.
    El generador de validación solo normaliza los píxeles al rango [0, 1] para
    evaluar el modelo sobre datos sin modificar.

    Retorna: (train_generator, validation_generator)
    """
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

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

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_gen, val_gen = get_data_generators(current_dir)
