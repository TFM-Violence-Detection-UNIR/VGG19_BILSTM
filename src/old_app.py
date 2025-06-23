import streamlit as st
import tempfile
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import imageio

load_dotenv()
BEST_MODEL = os.getenv("BEST_MODEL")

@st.cache_resource
def load_violence_model(path=BEST_MODEL):
    return load_model(path)

model = load_violence_model()

st.set_page_config(page_title="Detector de Violencia en Vídeo", layout="wide")

def convert_to_mp4(input_path):
    """
    Convierte un vídeo AVI u otro formato no soportado a MP4 usando imageio-ffmpeg (sin necesidad de ffmpeg instalado globalmente).
    """
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

    cap.release()
    writer.release()
    return temp_mp4.name

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        st.error("El vídeo es demasiado corto para realizar la detección.")
        cap.release()
        return None, video_path

    indices = np.linspace(0, total_frames-1, 5, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            st.error(f"No se pudo leer el frame {idx} de {video_path}")
            break
        frame = cv2.resize(frame, (224,224))
        frame = np.expand_dims(frame, axis=0)
        frames.append(frame)

    cap.release()
    if len(frames) != 5:
        return None, video_path

    return np.array(frames), video_path


def render_videos_grid(video_paths):
    """
    Muestra una lista de vídeos en una malla de hasta 3 columnas.
    """
    cols = st.columns(3)
    for idx, path in enumerate(video_paths):
        col = cols[idx % 3]
        with col:
            # Si no está en MP4, convertir temporalmente
            ext = os.path.splitext(path)[1].lower()
            show_path = path
            if ext in ['.avi', '.mov', '.wmv']:
                show_path = convert_to_mp4(path)
            st.video(show_path)


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


def handle_local_videos():
    base_dir = os.path.join(os.getcwd(), "datasets", "hockey_fights", "test")
    options, paths = [], {}
    for label in ["fight", "no_fight"]:
        folder = os.path.join(base_dir, label)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower().endswith((".mp4", ".avi", ".mov")):
                    rel = os.path.join(label, f)
                    options.append(rel)
                    paths[rel] = os.path.join(folder, f)

    selected = st.multiselect("Selecciona uno o varios vídeos", options)
    video_list = []
    if selected:
        for sel in selected:
            video_list.append(paths[sel])
        render_videos_grid(video_list)
    return video_list


def detect_violence(videos):
    for video in videos:
        data, vp = process_video(video)
        if data is None:
            continue
        probs = model.predict(data)[0]
        cls = np.argmax(probs)
        label = "Violencia" if cls == 1 else "No violencia"
        st.markdown(
            f"**Vídeo:** {vp}  \n"
            f"**Resultado:** {label}  \n"
            f"**Probabilidad No violencia:** {probs[0]:.2%}  \n"
            f"**Probabilidad Violencia:** {probs[1]:.2%}"
        )

# ----------------------
# Inicio de la aplicación
# ----------------------

st.title("Detector de Violencia en Vídeo")
st.write("Sube un vídeo o selecciona uno o varios vídeos locales y el modelo te dirá si contienen violencia o no.")

source = st.radio(
    "Selecciona la fuente de vídeo",
    ["Subir vídeo(s)", "Seleccionar vídeo(s) locales"]
)
videos = []

if source == "Subir vídeo(s)":
    videos = handle_uploaded_videos()
elif source == "Seleccionar vídeo(s) locales":
    videos = handle_local_videos()

if st.button("Detectar violencia"):
    if videos:
        detect_violence(videos)
    else:
        st.error("No se han seleccionado vídeos.")
