import os
import matplotlib.pyplot as plt

from data_prep import get_data_generators
from cnn_basica import build_simple_cnn
from transfer_learning import build_vgg16_transfer_learning

def plot_training_history(history_cnn, history_vgg, epochs):
    """
    Genera y guarda una figura comparativa con dos subplots.

    Subplot izquierdo: accuracy de entrenamiento y validación de ambos modelos por época.
    Subplot derecho: loss de entrenamiento y validación de ambos modelos por época.
    La figura se guarda en disco como 'comparacion_modelos.png'.
    """
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_cnn.history['accuracy'], label='CNN Básica Train Acc', linestyle='--')
    plt.plot(epochs_range, history_cnn.history['val_accuracy'], label='CNN Básica Val Acc', marker='o')
    plt.plot(epochs_range, history_vgg.history['accuracy'], label='VGG16 Train Acc', linestyle='--')
    plt.plot(epochs_range, history_vgg.history['val_accuracy'], label='VGG16 Val Acc', marker='^')
    plt.title('Comparación de Accuracy (Entrenamiento vs Validación)')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_cnn.history['loss'], label='CNN Básica Train Loss', linestyle='--')
    plt.plot(epochs_range, history_cnn.history['val_loss'], label='CNN Básica Val Loss', marker='o')
    plt.plot(epochs_range, history_vgg.history['loss'], label='VGG16 Train Loss', linestyle='--')
    plt.plot(epochs_range, history_vgg.history['val_loss'], label='VGG16 Val Loss', marker='^')
    plt.title('Comparación de Loss (Entrenamiento vs Validación)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('comparacion_modelos.png')
    print("\n[+] Gráfica guardada exitosamente como 'comparacion_modelos.png'")
    plt.show()

def main():
    """
    Ejecuta el pipeline completo: carga datos, entrena ambos modelos y genera las gráficas.

    Pasos:
    (1) preparar generadores de datos,
    (2) entrenar CNN básica y guardarla,
    (3) entrenar VGG16 y guardarla, (4) graficar y comparar los historiales.
    """
    print("=== PASO 1: Preparando Datos ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_gen, val_gen = get_data_generators(current_dir, target_size=(150, 150), batch_size=32)

    EPOCHS = 15

    print("\n=== PASO 2: Entrenando CNN Básica ===")
    modelo_cnn = build_simple_cnn()
    history_cnn = modelo_cnn.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    modelo_cnn.save('modelo_cnn_basica.h5')

    print("\n=== PASO 3: Entrenando Transfer Learning (VGG16) ===")
    modelo_vgg = build_vgg16_transfer_learning()
    history_vgg = modelo_vgg.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    modelo_vgg.save('modelo_vgg16.h5')

    print("\n=== PASO 4: Generando Gráficas de Resultados ===")
    plot_training_history(history_cnn, history_vgg, EPOCHS)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nEntrenamiento cancelado por el usuario.")
    except Exception as e:
        print(f"\nOcurrió un error: {e}")
