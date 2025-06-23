import streamlit as st
import tempfile
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import imageio

# Configuración de la página
st.set_page_config(page_title="Detector de Violencia en Vídeo", layout="wide")

# Modelos y mapeo a carpetas de datasets
MODEL_FILES = [
    "best_model_hockey_fights.keras",
    "best_model_rlvs.keras",
    "best_model_rwf.keras",
    "best_model_violent_flow.keras"
]
# Carpeta exacta donde están los datasets
# Los nombres deben coincidir con las carpetas en ./datasets
MODEL_TO_DATASET = {
    "best_model_hockey_fights.keras": "hockey_fights",
    "best_model_rlvs.keras":       "RLVS",
    "best_model_rwf.keras":        "RWF-800",
    "best_model_violent_flow.keras":"violent-flow"
}
DATASETS = list(MODEL_TO_DATASET.values())

def sync_dataset():
    """
    Sincroniza la selección de dataset con el modelo elegido.
    """
    st.session_state.dataset = MODEL_TO_DATASET[st.session_state.model]

# Sidebar: selección de modelo y dataset
model_choice = st.sidebar.selectbox(
    "Selecciona el modelo", MODEL_FILES,
    key="model",
    on_change=sync_dataset
)
dataset_choice = st.sidebar.selectbox(
    "Selecciona el dataset", DATASETS,
    key="dataset"
)

# Ruta al modelo seleccionado
model_path = os.path.join("models", model_choice)

@st.cache_resource
def load_violence_model(path):
    """
    Carga y cachea el modelo de detección de violencia.
    """
    return load_model(path)

# Cargamos el modelo seleccionado
model = load_violence_model(model_path)

# Función para convertir vídeos a MP4 si es necesario
def convert_to_mp4(input_path):
    try:
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data().get('fps', 24)
    except Exception as e:
        st.error(f"No se pudo leer el vídeo para conversión: {e}")
        return input_path

    temp_mp4 = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    try:
        writer = imageio.get_writer(
            temp_mp4.name,
            format='ffmpeg',
            mode='I',
            fps=fps,
            codec='libx264',
            ffmpeg_params=['-preset', 'fast', '-movflags', '+faststart']
        )
        for frame in reader:
            writer.append_data(frame)
    except Exception as e:
        st.error(f"Error durante la conversión a MP4: {e}")
        return input_path
    finally:
        writer.close()
        reader.close()

    return temp_mp4.name

# Función para extraer 10 frames espaciados uniformemente
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        st.error("El vídeo es demasiado corto para realizar la detección.")
        cap.release()
        return None, video_path

    indices = np.linspace(0, total_frames-1, 10, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            st.error(f"No se pudo leer el frame {idx} de {video_path}")
            break
        frame = cv2.resize(frame, (224,224))
        # frame = np.expand_dims(frame, axis=0)
        frames.append(frame)

    cap.release()
    if len(frames) != 10:
        return None, video_path

    # Apilamos y añadimos dimensión de batch: (1, 10, 224, 224, 3)
    data = np.stack(frames, axis=0)
    data = np.expand_dims(data, axis=0)
    return data, video_path

# Función para mostrar vídeos en grilla de hasta 3 columnas
def render_videos_grid(video_paths):
    cols = st.columns(min(3, len(video_paths)))
    for idx, path in enumerate(video_paths):
        col = cols[idx % 3]
        with col:
            ext = os.path.splitext(path)[1].lower()
            show_path = convert_to_mp4(path) if ext in ['.avi', '.mov', '.wmv'] else path
            st.video(show_path)

# Manejo de vídeos subidos por el usuario
def handle_uploaded_videos():
    video_list = []
    uploaded_files = st.file_uploader(
        "Elige uno o varios vídeos (mp4, avi, mov)",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for vf in uploaded_files:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vf.name)[1])
            tfile.write(vf.read())
            tfile.flush()
            video_list.append(tfile.name)
        render_videos_grid(video_list)
    return video_list

# Manejo de vídeos locales según el dataset seleccionado
def handle_local_videos(dataset):
    base_dir = os.path.join(os.getcwd(), "datasets", dataset, "test")
    options = {}
    # Recorrer subcarpetas (etiquetas) dinámicamente
    if os.path.isdir(base_dir):
        for label in os.listdir(base_dir):
            folder = os.path.join(base_dir, label)
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith((".mp4", ".avi", ".mov")):
                        key = f"{label}/{f}"
                        options[key] = os.path.join(folder, f)
    selected = st.multiselect("Selecciona uno o varios vídeos", list(options.keys()))
    video_list = [options[s] for s in selected]
    if video_list:
        render_videos_grid(video_list)
    return video_list

# Detección de violencia con visualización gráfica
def detect_violence(videos):
    results = []
    for video in videos:
        data, vp = process_video(video)
        if data is None:
            continue
        probs = model.predict(data)[0]
        results.append({"path": video, "probs": probs})

    # Mostrar solo la barra de probabilidad debajo de cada vídeo previamente renderizado
    cols = st.columns(len(results))
    for col, res in zip(cols, results):
        with col:
            prob = float(res['probs'][1])
            percent = int(prob * 100)
            st.write(f"**Probabilidad de violencia:** {prob:.2%}")
            st.progress(percent)

# ----- Inicio de la aplicación -----
st.title("Detector de Violencia en Vídeo")
st.write("Sube un vídeo o selecciona uno o varios vídeos locales del dataset y el modelo te dirá si contienen violencia o no.")

source = st.radio(
    "Selecciona la fuente de vídeo",
    ["Subir vídeo(s)", "Seleccionar vídeo(s) locales"]
)
videos = []
if source == "Subir vídeo(s)":
    videos = handle_uploaded_videos()
elif source == "Seleccionar vídeo(s) locales":
    videos = handle_local_videos(dataset_choice)

if st.button("Detectar violencia"):
    if videos:
        detect_violence(videos)
    else:
        st.error("No se han seleccionado vídeos.")
