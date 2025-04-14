import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,\
                            recall_score, f1_score, roc_curve, roc_auc_score
from keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model

def preprocess_frame(frame):
    # Preprocesamiento de la imagen
    frame = cv2.resize(frame, (224, 224))
    return np.expand_dims(frame, axis=0)

def loop_each_video(video_folder, target_frames):
    all_preprocessed_videos = []
    video_names = []
    # Iterar sobre cada video de entrenamiento en la carpeta
    for filename in os.listdir(video_folder):
        video_names.append(filename)
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
    return all_preprocessed_videos, video_names

def preprocess_testing_videos(fight_dir, non_fight_dir, sequence_length):
    # Preprocesar videos de violencia y no violencoa
    X_test_fight, fight_video_names = loop_each_video(fight_dir, sequence_length)
    X_test_non_fight, nonfight_video_names = loop_each_video(non_fight_dir, sequence_length)

    # Crear etiquetas para los datos de entrenamiento (1 para pelea, 0 para no pelea)
    y_test_fight = np.ones(len(X_test_fight))
    y_test_non_fight = np.zeros(len(X_test_non_fight))

    # Concatenar fight y nonfight
    X_test = np.concatenate((X_test_fight, X_test_non_fight))
    y_test = np.concatenate((y_test_fight, y_test_non_fight))
    y_test= to_categorical(y_test, num_classes=2)

    # Concatenar los nombres de los videos de fight y non fight
    all_video_names = fight_video_names + nonfight_video_names
    return X_test, y_test, all_video_names

def evaluate_model(model, X_test, y_test, all_video_names, folder, elapsed_time):
    # Hacer predicciones en los datos de prueba
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)

    # Calcular TP, TN, FP, FN en valores absolutos y porcentajes
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    total_videos = TP + TN + FP + FN

    TP_percentage = TP / total_videos * 100
    TN_percentage = TN / total_videos * 100
    FP_percentage = FP / total_videos * 100
    FN_percentage = FN / total_videos * 100

    # Almacenar nombres de los falsos positivos y de los falsos negativos.
    # Calcular Falsos Negativos (FN) y Falsos Positivos (FP)
    fn_indices = np.where((predicted_classes == 0) & (np.argmax(y_test, axis=1) == 1))[0]
    fp_indices = np.where((predicted_classes == 1) & (np.argmax(y_test, axis=1) == 0))[0]

    # Obtener nombres de videos para Falsos Negativos y Falsos Positivos
    fn_videos = [all_video_names[i] for i in fn_indices]
    fp_videos = [all_video_names[i] for i in fp_indices]

    # Calcular métricas
    accuracy = accuracy_score(np.argmax(y_test, axis=1), predicted_classes)
    precision = precision_score(np.argmax(y_test, axis=1), predicted_classes)
    recall = recall_score(np.argmax(y_test, axis=1), predicted_classes)
    f1 = f1_score(np.argmax(y_test, axis=1), predicted_classes)

    # Calcular especificidad
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    # Calcular el área bajo la curva ROC (AUC)
    auc = roc_auc_score(y_test[:, 1], predictions[:, 1])
    fpr, tpr, _ = roc_curve(y_test[:, 1], predictions[:, 1])

    # Crear un DataFrame con los resultados
    data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 
                   'F1 Score', 'Specificity', 'AUC', 
                   'TP', 'TN', 'FP', 'FN',
                   'TP Percentage', 'TN Percentage', 'FP Percentage', 'FN Percentage', 
                   'Average Prediction Time', 'FN video name', 'FP video name'],
        'Value': [accuracy, precision, recall, f1, specificity, auc, 
                  TP, TN, FP, FN, TP_percentage, TN_percentage,
                  FP_percentage, FN_percentage, elapsed_time / len(X_test),
                  fn_videos, fp_videos]
    }

    results_df = pd.DataFrame(data)

    # Almacenar los valores de la curva ROC en el DataFrame existente
    results_df['FPR'] = pd.Series(fpr)
    results_df['TPR'] = pd.Series(tpr)

    # Exportar el DataFrame a un archivo CSV
    results_df.to_csv(os.path.join(folder, 'testing_results.csv'), index=False)

    return results_df

# MAIN

# ¡¡¡IMPORTANTE!!! modificar acorde al caso: sequence_length
#                                            test_fight_dir, test_non_fight_dir, folder
# En principio tiene sentido que AUC y specificity sean 0. Chequear cuando más datos.

# PARAMETROS Y DIRECTORIOS
# Definir directorios de entrenamiento
test_fight_dir = r"C:\Users\ADMIN\Documents\xai_vd\Hockey fights\test_test\fight"
test_non_fight_dir = r"C:\Users\ADMIN\Documents\xai_vd\Hockey fights\test_test\nonfight"
sequence_length = 40  # Longitud de la secuencia temporal para la LSTM
folder = r"C:\Users\ADMIN\Documents\xai_vd\trained_models\join_vgg_lstm"

# PREPROCESADO DEL INPUT DEL ALGORITMO. X_test:(4, 40, 224, 224, 3), y_test: (4,2)
X_test, y_test, all_video_names = preprocess_testing_videos(test_fight_dir, test_non_fight_dir, sequence_length)

# IMPORTAR EL MODELO
model_path = os.path.join(folder, "best_model.keras")
loaded_model = load_model(model_path)

# HACER PREDICCIONES CON EL MODELO
start_time = time.time()
predictions = loaded_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
end_time = time.time()

# EVALUAR EL MODELO
#FALTA ESTADÍSTICA SOBRE TIEMPO DE EJECUCIÓN DE LOS VIDEOS -> APORTAR ESTADIST INDICA LA RAPIDEZ DEL MODELO
evaluation_results = evaluate_model(loaded_model, X_test, y_test, all_video_names, folder, round(end_time - start_time, 2))