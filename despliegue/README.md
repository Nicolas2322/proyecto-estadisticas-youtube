# Despliegue en la nube

- `space/app.py`: servicio de Hugging Face Spaces (Gradio): interfaz web, API `/predecir`, análisis de un video por enlace (API de datos de YouTube, opcional con el secreto `YOUTUBE_API_KEY`), tablero de monitoreo de latencia y deriva, y ficha del modelo.
- `../notebooks/Despliegue_Hugging_Face.ipynb`: entrena los modelos finales en Colab (GPU T4), los publica en un repositorio de modelos de Hugging Face y crea o actualiza el Space con versiones de librerías fijadas.

El servicio descarga los modelos del repositorio indicado en la variable `MODEL_REPO`, de modo que una nueva versión del modelo se publica sin cambiar el código y se puede volver a la anterior desde el historial de revisiones.
