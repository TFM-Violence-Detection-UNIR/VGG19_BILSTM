from tensorflow.keras.layers import GlobalAveragePooling2D, TimeDistributed, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import Input, Bidirectional, Concatenate, AveragePooling2D, Flatten, GlobalAveragePooling1D
from keras.utils import plot_model


def plot_model_architecture(model, filename='model_architecture.png'):
    """
    Genera un gráfico de la arquitectura del modelo y lo guarda como imagen.

    Args:
        model (Model): Modelo Keras a graficar.
        filename (str, opcional): Nombre del archivo de salida para la imagen. Por defecto es 'model_architecture.png'.
    """
    plot_model(model, to_file=filename, 
               show_shapes=True, 
               show_layer_names=True, 
               show_layer_activations=True, 
               show_trainable=True,
               expand_nested=True,)
    print(f"Modelo guardado como {filename}")

def create_compile_model(sequence_length, lstm_units=(128, 128), dense_units=(128, 128), frozen_layers=5, learning_rate=0.01):
    """
    Construye y compila un modelo híbrido MobileNetV2 + LSTM para detección binaria.

    Args:
        sequence_length (int): Número de fotogramas en cada secuencia de entrada.
        lstm_units (tuple[int, int], opcional): Unidades en las dos capas LSTM (layer1, layer2).
        dense_units (tuple[int, int], opcional): Unidades en las dos capas Dense intermedias.
        frozen_layers (int, opcional): Número de capas iniciales de MobileNetV2 a congelar.
        learning_rate (float, opcional): Tasa de aprendizaje para el optimizador Adam.

    Returns:
        Sequential: Modelo compilado listo para entrenar.
    """
    input_layer = Input(shape=(sequence_length, 224, 224, 3)) #(40 frames por vídeo), cada uno de tamaño 224x224x3 (color)
    # Rama A: CNN (MobileNetV2) hacia Bi-LSTM
    cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #Se carga MobileNet preentrenado con ImageNet sin las capas de clasificación (include_top=False).
    # cnn = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #Se carga VGG16 preentrenado con ImageNet sin las capas de clasificación (include_top=False).
    cnn.trainable = False #Se congela (trainable = False) para no reentrenar sus pesos, lo que ahorra tiempo y evita sobreajuste.
    cnn_branch = TimeDistributed(cnn)(input_layer) #TimeDistributed aplica la CNN a cada frame individualmente.
    cnn_branch = TimeDistributed(GlobalAveragePooling2D())(cnn_branch) #Luego se aplica GlobalAveragePooling2D por frame → convierte cada mapa de características en un vector.

    # Rama B: "bruto" hacia Bi-LSTM
    # Primero reducimos el frame bruto sin CNN
    pooled = TimeDistributed(AveragePooling2D(pool_size=(8, 8)))(input_layer) #Aplica AveragePooling2D(8x8) a cada frame → reduce 224×224 → 28×28 (más manejable).
    flattened = TimeDistributed(Flatten())(pooled)  # (28,28,3) → vector. Convierte cada frame (28×28×3) en un vector plano: 2.352 valores por frame.
    x_lstm = Bidirectional(LSTM(lstm_units[0],
                                return_sequences=True))(flattened) #La secuencia de vectores compactos se le pasa a la BiLSTM. La salida es un vector de tamaño 256 (128 en cada dirección).
    lstm_branch = Bidirectional(LSTM(lstm_units[1]))(x_lstm)
    # Concatenar ambas ramas
    cnn_summary = GlobalAveragePooling1D()(cnn_branch)#Calcula el promedio temporal de todos los vectores de salida del CNN.
    merged = Concatenate()([cnn_summary, lstm_branch])#Se fusionan las salidas de ambas ramas (512 + 256 → 768).

    x = Dense(dense_units[0], activation='relu')(merged) #Capa densa intermedia con 128 neuronas y activación ReLU.
    x = Dense(dense_units[1], activation='relu')(x) #Capa densa intermedia con 64 neuronas y activación ReLU.
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # Ejemplo de uso
    model = create_compile_model(sequence_length=10, lstm_units=(128, 128))
    plot_model_architecture(model, filename='model_architecture.png')
    print("Modelo creado y guardado correctamente.")