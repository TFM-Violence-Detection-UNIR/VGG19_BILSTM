# Detección de Violencia en Video con IA

Este proyecto utiliza técnicas avanzadas de inteligencia artificial para la detección de violencia en videos, combinando el poder de redes neuronales convolucionales (CNN) y modelos de memoria a largo plazo (LSTM). El objetivo es desarrollar un sistema eficiente y preciso que pueda identificar comportamientos violentos en secuencias de video.

## Datasets
Los datasets se pueden encontrar en esta [direccion](https://alumnosunir-my.sharepoint.com/:f:/g/personal/mario_sanz482_comunidadunir_net/EroNibQNp1BHniQbifBzJNMBPWy4v9GSofUlA7PbPZgehQ) 

## Tecnologías Utilizadas

- **VGG19**: Red neuronal convolucional preentrenada para la extracción de características visuales.
- **BiLSTM**: Modelo bidireccional de memoria a largo plazo para el análisis de secuencias temporales.
- **Python**: Lenguaje principal para el desarrollo del proyecto.
- **TensorFlow/Keras**: Frameworks para la construcción y entrenamiento de modelos de IA.
- **OpenCV**: Biblioteca para el procesamiento de video.
- **NumPy y Pandas**: Herramientas para la manipulación y análisis de datos.

## Instalación

1. **Clona este repositorio**:
    ```bash
    git clone https://github.com/TFM-Violence-Detection-UNIR/VGG19_BILSTM.git
    cd VGG19_BILSTM
    ```

2. **Creación y activación del entorno virtual**

    - **Windows**
      1. Crea el entorno virtual:
          ```bash
          python -m venv venv
          ```
      2. Activa el entorno:
          ```bash
          venv\Scripts\activate
          ```

    - **Linux/macOS**
      1. Crea el entorno virtual:
          ```bash
          python3 -m venv venv
          ```
      2. Activa el entorno:
          ```bash
          source venv/bin/activate
          ```

2. **Instalación de las dependencias**
    - Una vez activado el entorno virtual, instala las dependencias:
      ```bash
      pip install -r requirements.txt
      ```

3. **Creación del archivo .env con las variables de entorno**
    - En la raíz del proyecto, crea un archivo llamado `.env` y añade las variables necesarias (rutas).
    - Asegúrate de configurar las variables según las necesidades de tu proyecto.



## Licencia


## Contacto
