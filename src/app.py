import streamlit as st
import tempfile
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()
BEST_MODEL = os.getenv("BEST_MODEL")

@st.cache_resource
def load_violence_model(path=BEST_MODEL):
    return load_model(path)

model = load_violence_model()

def process_video(video_path):
    """_summary_

    Args:
        video_path (_type_): _description_
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        st.error("El vídeo es demasiado corto para realizar la detección.")
        cap.release()
        return None
    
    # Obtiene 5 índices equidistantes para los frames
    indices = np.linspace(0, total_frames-1, 5, dtype=int)
    preprocessed_video = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            st.error(f"No se pudo leer el frame {idx} de {video_path}")
            break
        # Redimensionar a 224x224, convertir BGR a RGB y normalizar
        frame = cv2.resize(frame, (224,224))
        frame = np.expand_dims(frame, axis=0)
        preprocessed_video.append(frame)
    preprocessed_video = np.array(preprocessed_video)
    cap.release()
    return preprocessed_video, video_path
    
def handle_uploaded_videos():
    """_summary_
    """
    video_list = []
    uploaded_files = st.file_uploader("Elige uno o varios vídeos", type=["mp4","avi","mov"], accept_multiple_files=True)
    if uploaded_files:
        for video_file in uploaded_files:
            # Escribir en un fichero temporal
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            tfile.flush()
            video_list.append(tfile.name)
            st.video(tfile.name)
    return video_list
    
def handle_local_videos():
    video_list = []
    # Ruta a la carpeta que contiene los vídeos locales
    base_dir = os.path.join(os.getcwd(), "datasets", "hockey_fights", "test")
    video_options = []
    video_paths = {}
    for label in ["fight", "no_fight"]:
        folder = os.path.join(base_dir, label)
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if file.lower().endswith((".mp4","avi","mov")):
                    rel_path = os.path.join(label, file)
                    full_path = os.path.join(folder, file)
                    video_options.append(rel_path)
                    video_paths[rel_path] = full_path

    selected_videos = st.multiselect("Selecciona uno o varios vídeos", video_options)
    if selected_videos:
        for option in selected_videos:
            path = video_paths[option]
            with open(path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            video_list.append(path)
    return video_list

def detect_violence(videos):
    """_summary_

    Args:
        videos (_type_): _description_
    """
    for video in videos:
        preprocessed_video, processed_video = process_video(video)
        if preprocessed_video is None or preprocessed_video.shape[0] != 5:
            continue
        
        # Inferencia
        probs = model.predict(preprocessed_video)[0]
        probs = model.predict(preprocessed_video)[0]  # Vector de 2 clases
        cls = np.argmax(probs)
        label = "Violencia" if cls == 1 else "No violencia"
        st.markdown(
            f"**Vídeo:** {processed_video}  \n"
            f"**Resultado:** {label}  \n"
            f"**Probabilidad No violencia:** {probs[0]:.2%}  \n"
            f"**Probabilidad Violencia:** {probs[1]:.2%}"
        )

# ------------------------------------------------------------------
# Inicio de la aplicación
# ------------------------------------------------------------------

st.title("Detector de Violencia en Vídeo")
st.write("Sube un vídeo o selecciona uno o varios vídeos locales y el modelo te dirá si contienen violencia o no.")

# Selección de fuente: vídeos subidos o vídeos locales
source = st.radio("Selecciona la fuente de vídeo", ["Subir vídeo(s)", "Seleccionar vídeo(s) locales"])
videos = []

if source == "Subir vídeo(s)":
    videos = handle_uploaded_videos()
elif source == "Seleccionar vídeo(s) locales":
    videos = handle_local_videos()

# Botón para ejecutar la detección siempre visible
if st.button("Detectar violencia"):
    if videos:
        detect_violence(videos)
    else:
        st.error("No se han seleccionado vídeos.")