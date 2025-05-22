import streamlit as st
import tempfile
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

from dotenv import load_dotenv
load_dotenv()
BEST_MODEL = os.getenv("BEST_MODEL")
# 1. Carga del modelo
@st.cache_resource
def load_violence_model(path=BEST_MODEL):
    return load_model(path)

model = load_violence_model()

st.title("Detector de Violencia en Vídeo")
st.write("Sube un vídeo y el modelo te dirá si contiene violencia o no.")

# 2. Carga del vídeo
video_file = st.file_uploader("Elige un vídeo", type=["mp4","avi","mov"])
if video_file:
    # Mostrar el vídeo
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    st.video(tfile.name)

    # Botón para procesar
    if st.button("Detectar violencia"):
        # 3. Leer vídeo y extraer 5 frames espaciados
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames-1, 5, dtype=int)

        preprocessed_video = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                st.error(f"No se pudo leer el frame {idx}")
                break
            # 224x224 + BGR→RGB + normalización [0,1]
            frame = cv2.resize(frame, (224,224))
            frame = np.expand_dims(frame, axis=0)
            preprocessed_video.append(frame)
        preprocessed_video = np.array(preprocessed_video)
        cap.release()

        # 4. Inferencia
        probs = model.predict(preprocessed_video)[0]  # vector 2 clases
        print(probs)
        cls = np.argmax(probs)
        conf = probs[cls]

        # 5. Mostrar resultado
        label = "Violencia" if cls==1 else "No violencia"
        
        st.markdown(f"**Resultado:** {label}  \n**Probabilidad No violencia:** {probs[0]:.2%}  \n**Probabilidad Violencia:** {probs[1]:.2%}")
