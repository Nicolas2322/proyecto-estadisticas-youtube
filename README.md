# Proyecto Estadísticas de YouTube — Procesamiento de Lenguaje Natural

**Universidad de Cundinamarca · Especialización en Inteligencia Artificial**
Asignatura: Procesamiento de Lenguaje Natural · Docente: Paul Alexander Díaz Montaña
Estudiante: Yohan Nicolás Gómez Castañeda

Objetivo: predecir el desempeño de un video de YouTube a partir de sus comentarios mediante procesamiento de lenguaje natural.

## Arquitectura (dos etapas)

```
Comentarios ──► Etapa A: sentimiento (SVM lineal, TF-IDF de palabras + caracteres)
            ──► Agregación por video (sentimiento, likes de comentarios, Gini, tópicos SVD)
            ──► Etapa B: XGBoost → log10(vistas)  |  Regresión logística → ¿éxito en su categoría?
```

## Resultados principales

| Tarea | Modelo seleccionado | Desempeño |
|---|---|---|
| Sentimiento por comentario | SVM lineal (palabras + caracteres), C = 0,1 | F1-macro 0,710 · exactitud 0,784 (VADER: 0,596 / 0,715) |
| log10(vistas) | XGBoost, 32 variables | R² 0,787 (CV) · 0,80 en partición reservada |
| Éxito relativo en su categoría | Regresión logística, C = 100 | ROC-AUC 0,873 (CV) · 0,890 en partición reservada |

- 16 algoritmos y 43 configuraciones comparadas con validación cruzada de 5 particiones.
- Búsqueda en malla (`GridSearchCV`) de 74 combinaciones de hiperparámetros.
- Curvas de aprendizaje para valorar la suficiencia de los datos.
- Los likes y comentarios del video se excluyen como predictores por fuga de información (correlación de 0,95 con las vistas).

## Estructura

```
notebooks/proyecto_youtube_avances_II.ipynb   Notebook ejecutado (compatible con Google Colab)
src/analisis.py                               Limpieza, estadística, modelos, ajuste y figuras
src/curvas.py                                 Curvas de aprendizaje
reports/                                      Informe PDF, resultados.json, curvas.json y figuras (f1–f7)
data/                                         Ubicación de los CSV de Kaggle (no versionados)
```

## Ejecución

```bash
pip install -r requirements.txt
python src/analisis.py data/videos-stats.csv data/comments.csv reports
python src/curvas.py   data/videos-stats.csv data/comments.csv reports
```

O bien abrir el notebook desde la carpeta `notebooks/`. Semilla fija (42) para reproducibilidad.

## Avances III: iteración del modelo

| Etapa | Base (Avances II) | Modelo final (Avances III) |
|---|---|---|
| A · Sentimiento | SVM palabras + caracteres · F1-macro 0,708 | RoBERTa-Twitter con ajuste fino · **F1-macro 0,900** (alternativa en CPU: SVM híbrido con GloVe y umbrales anidados · 0,757) |
| B · log10(vistas) | XGBoost · R² 0,787 | Stacking XGB + HGB + RF + Ridge (meta RidgeCV) · **R² 0,793**, intervalos conformales del 90 % |
| Éxito relativo | Regresión logística · AUC 0,873 | Regresión logística · **AUC 0,879** |

Limpieza documentada en `reports/avances3/registro_limpieza/` (conteos antes/después, huellas SHA-256 y respaldo de los registros retirados para poder revertir cada paso).

25 iteraciones registradas en MLflow (`reports/avances3/mlflow_runs.csv`), con Optuna (CV anidada), SHAP y pruebas de McNemar y Wilcoxon.

```bash
pip install -r requirements.txt optuna mlflow shap gensim emoji
python src/avances3/registro_limpieza.py data/videos-stats.csv data/comments.csv reports/avances3/registro_limpieza
python src/avances3/iter_a.py data/videos-stats.csv data/comments.csv reports/avances3
python src/avances3/iter_b.py reports/features_video.pkl reports/avances3 data/comments.csv
python src/avances3/figuras.py reports/avances3
```

`notebooks/A6_transformer_colab.ipynb`: iteración con RoBERTa-Twitter (requiere GPU en Google Colab).

## Entregas

- [x] Avances II (semana 8): datos, análisis estadístico, modelos, métricas, ajuste y selección de variables — [`reports/Informe_Avances_II_Gomez_Castaneda.pdf`](reports/Informe_Avances_II_Gomez_Castaneda.pdf)
- [x] Avances III (semana 9): iteración y mejora del modelo — [`reports/avances3/Informe_Avances_III_Gomez_Castaneda.pdf`](reports/avances3/Informe_Avances_III_Gomez_Castaneda.pdf)
- [ ] Entrega final (semana 10): API desplegada en la nube y monitoreo
