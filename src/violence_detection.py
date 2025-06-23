import os
import cv2
import csv
import datetime
import numpy as np
import pandas as pd
import random
import shutil

import matplotlib.pyplot as plt

from keras.callbacks import CSVLogger, EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D, TimeDistributed, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Concatenate, Bidirectional, AveragePooling2D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
# Variables de entorno
from dotenv import load_dotenv
load_dotenv()

RESULTS_ROUTE = os.getenv("RESULTS_ROUTE")
HOCKEY_ROUTE = os.getenv("HOCKEY_ROUTE")
HOCKEY_TRAIN_FIGHT_ROUTE = os.getenv("HOCKEY_TRAIN_FIGHT_ROUTE")
HOCKEY_TRAIN_NOFIGHT_ROUTE = os.getenv("HOCKEY_TRAIN_NOFIGHT_ROUTE")


def save_folder(route):
    """
    Genera y crea, si es necesario, una nueva carpeta en el directorio especificado, 
    utilizando la fecha y hora actual para formar su nombre.
    Args:
        route (str): Ruta base donde se guardará la carpeta.
    Returns:
        str: Ruta completa de la carpeta creada o existente.
    Notas:
        - El nombre de la carpeta se genera concatenando el día, el nombre abreviado del mes 
        y la hora en formato HHMM.
        - Si la carpeta ya existe, no se crea nuevamente.
    """
    
    now = datetime.datetime.now()
    number_day_month = now.strftime("%d").lower()
    month_name = now.strftime("%b").lower()
    hour = now.strftime("%H%M")
    
    folder = route + "\\" + number_day_month + month_name + hour
    
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as error:
            pass
        
    return folder


def split_data(root_dir, train_pct=0.8, seed=None):
    """
    Splits the contents of each class-folder under `root_dir` into train and test sets.

    Args:
        root_dir (str): Path to your dataset folder (e.g. "datasets/hockey_fights").
                        Inside this folder you should have one folder per class
                        (e.g. "fight", "no_fight").
        train_pct (float): Fraction of data to allocate to training (0 < train_pct < 1).
                           The remainder (1 - train_pct) goes to testing.
        seed (int, optional): Random seed for reproducibility. If None, shuffle is random.

    After running, you’ll end up with:
        root_dir/
            fight/        # original data untouched
            no_fight/     # original data untouched
            train/
                fight/
                no_fight/
            test/
                fight/
                no_fight/
    """
    if not 0 < train_pct < 1:
        raise ValueError("train_pct must be between 0 and 1")

    if seed is not None:
        random.seed(seed)

    # Discover class subfolders (ignore any existing train/test directories)
    classes = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d not in ("train", "test")
    ]

    for cls in classes:
        src_dir   = os.path.join(root_dir, cls)
        train_dir = os.path.join(root_dir, "train", cls)
        test_dir  = os.path.join(root_dir, "test", cls)

        # Create train/test dirs if needed
        for d in (train_dir, test_dir):
            if not os.path.exists(d):
                os.makedirs(d)
                print(f"Created directory {d}")
            else:
                print(f"Directory {d} already exists; skipping creation.")

        # List and shuffle files
        files = [
            f for f in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, f))
        ]
        random.shuffle(files)

        split_idx = int(len(files) * train_pct)
        train_files = files[:split_idx]
        test_files  = files[split_idx:]

        # Copy files
        for fname in train_files:
            shutil.copy2(os.path.join(src_dir, fname),
                         os.path.join(train_dir, fname))
        for fname in test_files:
            shutil.copy2(os.path.join(src_dir, fname),
                         os.path.join(test_dir, fname))

        print(f"{cls}: {len(train_files)} → train, {len(test_files)} → test")
        

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocesa un único fotograma para que tenga el tamaño y la forma adecuados
    para la entrada de una red neuronal.

    Args:
        frame (np.ndarray): Fotograma original en formato BGR (altura, anchura, canales).
        target_size (tuple[int, int], optional): Tamaño al que redimensionar
            el fotograma (anchura, altura). Por defecto (224, 224).

    Returns:
        np.ndarray: Fotograma redimensionado a (1, target_height, target_width, 3), listo para batch.
    """
    # Redimensionar el fotograma al tamaño deseado
    frame = cv2.resize(frame, target_size)
    # Añadir la dimensión de batch
    return np.expand_dims(frame, axis=0)

def loop_each_video(video_folder, target_frames=30, frame_size=(224, 224)):
    """
    Recorre todos los vídeos .avi en una carpeta, extrae un número fijo de
    fotogramas equidistantes de cada uno, y los preprocesa.

    Args:
        video_folder (str): Ruta a la carpeta que contiene los archivos .avi.
        target_frames (int): Número de fotogramas a extraer y preprocesar por vídeo.
        frame_size (tuple[int, int], optional): Tamaño al que redimensionar
            cada fotograma (anchura, altura). Por defecto (224, 224).

    Returns:
        np.ndarray: Array de forma (num_vídeos, target_frames, frame_height, frame_width, 3)
                    con todos los fotogramas preprocesados de cada vídeo.
    """
    all_preprocessed_videos = []

    # Iterar sobre cada video de entrenamiento en la carpeta
    for filename in os.listdir(video_folder):
        if filename.endswith(".avi"):  # Asegúrate de que estás procesando archivos de video
            video_path = os.path.join(video_folder, filename)
            cap = cv2.VideoCapture(video_path)

            # Calcular el número total de fotogramas en el video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Truncar equidistantemente al número deseado de fotogramas
            indices_truncados = np.linspace(0, total_frames-1, target_frames, dtype=int)

            # Lista para almacenar las características de cada fotograma
            preprocessed_video = []

            # Iterar sobre cada fotograma
            for i in indices_truncados:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Solo seleccionara los frames seleccionados equidistantes
                ret, frame = cap.read()
                if not ret:
                    break
                preprocessed_frame = preprocess_frame(frame, target_size=frame_size)
                preprocessed_video.append(preprocessed_frame)

            preprocessed_video = np.array(preprocessed_video)
            all_preprocessed_videos.append(np.squeeze(preprocessed_video))
            # Liberar el objeto de captura
            cap.release()

    all_preprocessed_videos = np.array(all_preprocessed_videos)
    return all_preprocessed_videos
  
def preprocess_training_videos(fight_dir, non_fight_dir, sequence_length, frame_size=(224, 224)):
    """
    Preprocesa los vídeos de entrenamiento de dos carpetas (peleas y no peleas),
    extrae fotogramas, genera etiquetas one-hot y mezcla los datos.

    Args:
        fight_dir (str): Carpeta con vídeos de peleas (etiqueta 1).
        non_fight_dir (str): Carpeta con vídeos sin peleas (etiqueta 0).
        sequence_length (int): Número de fotogramas a extraer de cada vídeo.
        frame_size (tuple[int, int], optional): Tamaño al que redimensionar
            cada fotograma (anchura, altura). Por defecto (224, 224).

    Returns:
        tuple:
            - X_train (np.ndarray): Datos de entrada con forma
              (num_total_vídeos, sequence_length, frame_height, frame_width, 3).
            - y_train (np.ndarray): Etiquetas one-hot con forma
              (num_total_vídeos, 2).
    """
    # Preprocesar videos de violencia y no violencoa
    X_train_fight = loop_each_video(fight_dir, sequence_length, frame_size)
    X_train_non_fight = loop_each_video(non_fight_dir, sequence_length, frame_size)

    # Crear etiquetas para los datos de entrenamiento (1 para pelea, 0 para no pelea)
    y_train_fight = np.ones(len(X_train_fight))
    y_train_non_fight = np.zeros(len(X_train_non_fight))

    # Concatenar fight y nonfight
    X_train = np.concatenate((X_train_fight, X_train_non_fight))
    y_train = np.concatenate((y_train_fight, y_train_non_fight))
    y_train_encoded = to_categorical(y_train, num_classes=2)
    # Mezclar de forma aleatoria X_train e y_train
    X_train, y_train = shuffle(X_train, y_train_encoded, random_state=42)
    return X_train, y_train



def create_compile_model(sequence_length, lstm_units=(64, 64), dense_units=(64, 64), frozen_layers=5, learning_rate=0.01):
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
    mobile = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #Se carga MobileNet preentrenado con ImageNet sin las capas de clasificación (include_top=False).
    # mobile.trainable = False #Se congela (trainable = False) para no reentrenar sus pesos, lo que ahorra tiempo y evita sobreajuste.
    cnn_branch = TimeDistributed(mobile)(input_layer) #TimeDistributed aplica la CNN a cada frame individualmente.
    cnn_branch = TimeDistributed(GlobalAveragePooling2D())(cnn_branch) #Luego se aplica GlobalAveragePooling2D por frame → convierte cada mapa de características en un vector.

    # Rama B: "bruto" hacia Bi-LSTM
    # Primero reducimos el frame bruto sin CNN
    pooled = TimeDistributed(AveragePooling2D(pool_size=(8, 8)))(input_layer) #Aplica AveragePooling2D(8x8) a cada frame → reduce 224×224 → 28×28 (más manejable).
    flattened = TimeDistributed(Flatten())(pooled)  # (28,28,3) → vector. Convierte cada frame (28×28×3) en un vector plano: 2.352 valores por frame.
    lstm_branch = Bidirectional(LSTM(128))(flattened) #La secuencia de vectores compactos se le pasa a la BiLSTM. La salida es un vector de tamaño 256 (128 en cada dirección).

    # Concatenar ambas ramas
    cnn_summary = GlobalAveragePooling1D()(cnn_branch)#Calcula el promedio temporal de todos los vectores de salida del CNN.
    merged = Concatenate()([cnn_summary, lstm_branch])#Se fusionan las salidas de ambas ramas (512 + 256 → 768).

    x = Dense(128, activation='relu')(merged) #Capa densa intermedia con 128 neuronas y activación ReLU.
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()  # Imprimir resumen del modelo
    return model # Se desea devolver el modelo, no la compilacion

def save_tuner_data(tuner, folder):
    """
    Extrae los detalles de cada trial del tuner y los guarda en un CSV.

    Args:
        tuner: Instancia de Keras-Tuner tras completar la búsqueda.
        folder (str): Carpeta donde se guardará `hyperparameters_trials_details.csv`.
    """
    # Obtener todos los ensayos realizados por el tuner
    trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))

    # Lista para almacenar los detalles de cada ensayo
    all_trials_details = []

    # Iterar sobre cada ensayo para obtener los detalles
    for trial in trials:
        trial_details = {
            'Trial ID': trial.trial_id,
            'LSTM Units Layer 1': trial.hyperparameters.values['lstm_units_layer1'],
            'LSTM Units Layer 2': trial.hyperparameters.values['lstm_units_layer2'],
            'Dense Units Layer 1': trial.hyperparameters.values['dense_units_layer1'],
            'Dense Units Layer 2': trial.hyperparameters.values['dense_units_layer2'],
            #'Frozen Layers': trial.hyperparameters.values['frozen_layers'],
            'Learning Rate': trial.hyperparameters.values['learning_rate'],
            'Score': trial.score
        }
        all_trials_details.append(trial_details)

    # Convertir los detalles de los ensayos a un DataFrame de pandas
    trials_df = pd.DataFrame(all_trials_details)

    # Guardar los detalles de los ensayos en un archivo CSV
    trials_df.to_csv(f"{folder}/hyperparameters_trials_details.csv", index=False)
    
def keras_tuner_train_model(X_train, y_train, folder, sequence_length):
    """
    Configura y ejecuta una búsqueda aleatoria de hiperparámetros con Keras-Tuner,
    guarda resultados y devuelve el mejor modelo.

    Args:
        X_train: Array de entrenamiento (secuencias de fotogramas).
        y_train: Etiquetas one-hot para clasificación binaria.
        folder (str): Ruta de directorio donde almacenar checkpoints y logs.
        sequence_length (int): Longitud de cada secuencia de fotogramas.

    Returns:
        Sequential: Mejor modelo encontrado por el tuner.
    """
    def build_model(hp):
        lstm_units_layer1 = hp.Choice('lstm_units_layer1', values=[128, 256, 512])
        lstm_units_layer2 = hp.Choice('lstm_units_layer2', values=[128, 256, 512])
        dense_units_layer1 = hp.Choice('dense_units_layer1', values=[128, 256, 512])
        dense_units_layer2 = hp.Choice('dense_units_layer2', values=[128, 256, 512])

        #frozen_layers = hp.Choice('frozen_layers', values=[3, 5, 8])

        learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])

        model = create_compile_model(sequence_length, 
                                     lstm_units=(lstm_units_layer1, lstm_units_layer2), 
                                     dense_units=(dense_units_layer1, dense_units_layer2),
                                     #frozen_layers=frozen_layers,
                                     learning_rate=learning_rate)
        return model

    tuner = RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=4,
            directory=folder,
            project_name=f'violence_detection'
        )

    checkpoint_path = os.path.join(folder, "training_1", "cp.weights.h5")
    #checkpoint_path = r"training_1\cp.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    X_train_partial, X_val, y_train_partial, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)    
    print("Train samples:", len(X_train_partial))
    print("Batch size:  ", sequence_length)
    import math
    print("=> steps/per epoch:", math.ceil(len(X_train_partial) / sequence_length))
    tuner.search(x=X_train_partial, 
                y=y_train_partial,
                epochs=2,
                batch_size=sequence_length,
                validation_data=(X_val, y_val),
                callbacks=[cp_callback, CSVLogger(os.path.join(folder, "tuner_results.csv"))])  # Agrega un callback para guardar los resultados en un archivo CSV)

    save_tuner_data(tuner, folder)

    # Obtener el mejor modelo y guardar
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

def train_best_model(X, y, folder, sequence_length):
    # Parámetros óptimos (ejemplo: obtenidos del trial #1)
    best_lstm_units = (256, 128)
    best_dense_units = (512, 256)
    best_learning_rate = 0.0001
    frozen_layers = 5

    # Crear modelo con hiperparámetros fijos
    model = create_compile_model(
        sequence_length=sequence_length,
        lstm_units=best_lstm_units,
        dense_units=best_dense_units,
        frozen_layers=frozen_layers,
        learning_rate=best_learning_rate
    )
    model.summary()

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Callbacks
    os.makedirs(folder, exist_ok=True)
    checkpoint_path = os.path.join(folder, "training_1", "cp.weights.h5")
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    log_path = os.path.join(folder, 'training_best.csv')
    csv_logger = CSVLogger(log_path)

    # Entrenamiento
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=5,                    # Ajusta el número de epochs
        batch_size=sequence_length,  # Cada lote = 1 vídeo (5 frames)
        validation_data=(X_val, y_val),
        callbacks=[cp_callback, csv_logger]
    )
    return model, history


def plot_history(history):
    """
        Esta función dibuja las gráficas de la precisión (accuracy) y la pérdida (loss) del modelo a lo largo de las épocas de entrenamiento.
        Se generan dos subgráficos: uno para visualizar la evolución de la precisión (tanto en entrenamiento como en validación) y otro para la evolución de la pérdida.
        Cada gráfico incluye anotaciones en el último punto de la curva para mostrar el valor final obtenido.
    Args:
        history (History): Objeto que contiene el historial del entrenamiento del modelo, el cual debe incluir las claves:
                           - 'accuracy': Precisión del entrenamiento por época.
                           - 'val_accuracy': Precisión de la validación por época.
                           - 'loss': Pérdida del entrenamiento por época.
                           - 'val_loss': Pérdida de la validación por época.
    Notas:
        - Se utiliza la función plt.tight_layout() para ajustar automáticamente la distribución de los subgráficos.
    """

    epochs = range(1, len(history.history['accuracy']) + 1)

    # Valores finales
    final_acc      = history.history['accuracy'][-1]
    final_val_acc  = history.history['val_accuracy'][-1]
    final_loss     = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax = axes[0]
    ax.plot(epochs, history.history['accuracy'], marker='o', markersize=4, linestyle='-', label='train_acc')
    ax.plot(epochs, history.history['val_accuracy'], marker='o', markersize=4, linestyle='-', label='val_acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.annotate(f"{final_acc:.3f}",     xy=(epochs[-1], final_acc),     xytext=(5, 5), textcoords="offset points")
    ax.annotate(f"{final_val_acc:.3f}", xy=(epochs[-1], final_val_acc), xytext=(5, -10), textcoords="offset points")
    
    #Loss
    ax = axes[1]
    ax.plot(epochs, history.history['loss'], marker='o', markersize=4, linestyle='-', label='train_loss')
    ax.plot(epochs, history.history['val_loss'], marker='o', markersize=4, linestyle='-', label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Model Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.annotate(f"{final_loss:.3f}",     xy=(epochs[-1], final_loss),     xytext=(5, 5), textcoords="offset points")
    ax.annotate(f"{final_val_loss:.3f}", xy=(epochs[-1], final_val_loss), xytext=(5, -10), textcoords="offset points")

    plt.tight_layout()
    plt.show()


#split_data(HOCKEY_ROUTE, train_pct=0.75, seed=42)

results_folder = save_folder(RESULTS_ROUTE)
sequence_length = 5
X_train, y_train = preprocess_training_videos(HOCKEY_TRAIN_FIGHT_ROUTE,
                                              HOCKEY_TRAIN_NOFIGHT_ROUTE, 
                                              sequence_length)
# Entrenamiento del modelo
# best_model = keras_tuner_train_model(X_train, y_train, results_folder, sequence_length)
best_model, history = train_best_model(X_train, y_train, results_folder, sequence_length)
plot_history(history)
# Guardar el mejor modelo
best_model.save(os.path.join(results_folder, "best_model.keras"))