import os
import csv
import cv2
import datetime
import numpy as np
import pandas as pd

from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D, TimeDistributed, LSTM, Dense
from tensorflow.keras.models import Sequential

def save_folder(route):
    # Eliminar la barra diagonal inversa al final de la ruta base si existe
    if route.endswith("\\"):
        route = route[:-1]

    # Obtener la fecha y hora actual
    now = datetime.datetime.now()
    numero_dia_mes = now.strftime("%d").lower()
    nombre_mes = now.strftime("%b").lower()
    hora = now.strftime("%H%M")

    folder = route + numero_dia_mes + nombre_mes + hora

    # Crear la carpeta si no existe
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as error:
            pass # No hagas nada específico en caso de error
    return folder

def preprocess_frame(frame):
    # Preprocesamiento de la imagen
    frame = cv2.resize(frame, (224, 224))
    return np.expand_dims(frame, axis=0)

def loop_each_video(video_folder, target_frames):
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
                preprocessed_frame = preprocess_frame(frame)
                preprocessed_video.append(preprocessed_frame)

            preprocessed_video = np.array(preprocessed_video)
            all_preprocessed_videos.append(np.squeeze(preprocessed_video))
            # Liberar el objeto de captura
            cap.release()

    all_preprocessed_videos = np.array(all_preprocessed_videos)
    return all_preprocessed_videos

def preprocess_training_videos(fight_dir, non_fight_dir, sequence_length):
    # Preprocesar videos de violencia y no violencoa
    X_train_fight = loop_each_video(fight_dir, sequence_length)
    X_train_non_fight = loop_each_video(non_fight_dir, sequence_length)

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
    # Importar VGG con los pesos de image_net sin incluir las capas finales de clasificacion
    vgg19_extract_features = VGG19(weights=None, include_top=False, input_shape=(224, 224, 3))
    # Congelar las primeras capas y descongelar las últimas
    # for layer in vgg19_extract_features.layers[:frozen_layers]:  # Congelar todas excepto las últimas 5 capas
    #     layer.trainable = False

    # Crear un modelo secuencial (adición de capas consecutivas)
    model = Sequential()
    # La capa pasa al modelo videos individuales como paquetes, dando cada frame por separado
    model.add(TimeDistributed(vgg19_extract_features, input_shape=(sequence_length, 224, 224, 3)))
    # Redimensionar la salida de TimeDistributed antes de pasarla a LSTM
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(LSTM(lstm_units[0], return_sequences=True))
    model.add(LSTM(lstm_units[1]))
    model.add(Dense(dense_units[0], activation='sigmoid'))
    model.add(Dense(dense_units[1], activation='sigmoid'))
    # Capa densa final con activación 'softmax' para un problema binario
    model.add(Dense(2, activation='sigmoid'))  # Ajusta el número de unidades según tu caso binario

    # Compilar el modelo
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model # Se desea devolver el modelo, no la compilacion

def save_tuner_data(tuner, folder):
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
            max_trials=1,
            directory=folder,
            project_name=f'violence_detection'
        )

    checkpoint_path = r"training_1\cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    X_train_partial, X_val, y_train_partial, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)    
    
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


# MAIN

# ¡¡¡IMPORTANTE!!! modificar acorde al caso: max_trials, 
#                                            epochs (un solo valor),
#                                            sequence_length
#                                            train_fight_dir, train_non_fight_dir, folder

# Por lo pronto no se implementa ninguna capa de Dropout o Regularización

# PARAMETROS Y DIRECTORIOS
# Definir directorios de entrenamiento
train_fight_dir = r"C:\Users\USUARIO\Desktop\2 CUATRI\TFM\VGG19_BILSTM\src\datasets\action-movies\fights"
train_non_fight_dir = r"C:\Users\USUARIO\Desktop\2 CUATRI\TFM\VGG19_BILSTM\src\datasets\action-movies\noFights"
sequence_length = 40  # Longitud de la secuencia temporal para la LSTM
folder = save_folder(r"C:\Users\USUARIO\Desktop\2 CUATRI\TFM\VGG19_BILSTM\src\trained-models\action-movies\\")

# PREPROCESADO DEL INPUT DEL ALGORITMO. X_train:(4, 40, 224, 224, 3), y_train: (4,2)
X_train, y_train = preprocess_training_videos(train_fight_dir, train_non_fight_dir, sequence_length)

print(y_train)
# # ENTRENAMIENTO DEL MODELO
# # AJUSTE DE HIPERPARAMETROS, OTRAS OPCIONES DE PARAMETROS, OTRAS OPCIONES DE LSTM Y CAPAS DENSAS.
# best_model = keras_tuner_train_model(X_train, y_train, folder, sequence_length)

# def save_model(folder, best_model):
#     best_model.save(os.path.join(folder, "best_model.keras"))


