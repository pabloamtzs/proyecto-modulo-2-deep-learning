# Análisis de Sentimientos en Tweets utilizando Deep Learning

Este proyecto tiene como objetivo desarrollar un modelo de deep learning para la clasificación de sentimientos en tweets, utilizando una Red Neuronal Recurrente (RNN) con capas LSTM (Long Short-Term Memory). El modelo es capaz de predecir si el sentimiento de un tweet es positivo o negativo, basándose en un conjunto de datos de tweets etiquetados.

## Objetivo

El objetivo de este proyecto es aplicar técnicas de procesamiento de lenguaje natural (NLP) y deep learning para entrenar un modelo que sea capaz de clasificar los sentimientos de los tweets en dos categorías: positivos y negativos. Para lograr esto, se empleó un conjunto de datos de tweets que contienen una variedad de opiniones expresadas por los usuarios en las redes sociales.

## Metodología

### 1. **Preprocesamiento de Datos**

El preprocesamiento es un paso crucial para preparar los datos y hacerlos adecuados para ser utilizados por un modelo de deep learning. Los pasos realizados en esta fase incluyen:

- **Limpieza de tweets**: Se creó una función para limpiar los tweets, eliminando enlaces web, menciones a usuarios (`@usuario`), hashtags, caracteres especiales y todo lo que no son letras o espacios. También se convirtió el texto a minúsculas y se eliminaron los espacios extra.
- **Tokenización y Padding**: El texto fue tokenizado, convirtiendo las palabras en secuencias numéricas. Luego, se aplicó padding para asegurar que todas las secuencias de tweets tengan la misma longitud.
- **Creación de conjuntos de datos**: Se generaron dos conjuntos de datos, `X` con las secuencias numéricas de los tweets y `y` con las etiquetas de sentimiento (0 para negativo y 1 para positivo).

### 2. **Desarrollo del Modelo**

El modelo utilizado para la clasificación de sentimientos es una **Red Neuronal Recurrente (RNN)** con capas LSTM. Los pasos y decisiones clave en el desarrollo del modelo fueron los siguientes:

- **Arquitectura del modelo**: El modelo tiene una capa **Embedding** que convierte las palabras en vectores numéricos densos. Luego, se agregaron dos capas LSTM con 128 y 64 unidades, respectivamente, para capturar relaciones contextuales en el texto. Se añadió una capa **Dropout** para evitar el sobreajuste.
- **Compilación y entrenamiento**: Se utilizó el optimizador **Adam** con la función de pérdida **binary_crossentropy** y la métrica **accuracy**. El modelo fue entrenado durante 5 épocas con un tamaño de lote de 64 y un 20% de los datos como conjunto de validación.

### 3. **Evaluación y Resultados**

Se evaluó el modelo utilizando un conjunto de test que no había sido visto durante el entrenamiento. Las métricas clave utilizadas para evaluar el rendimiento del modelo incluyen:

- **Exactitud (Accuracy)**: Proporción de predicciones correctas.
- **Precisión (Precision)**: Proporción de predicciones positivas correctas sobre todas las predicciones positivas.
- **Recuperación (Recall)**: Capacidad del modelo para identificar todos los tweets positivos.
- **F1-Score**: Combina la precisión y la recuperación en una única métrica.

Además, se realizaron predicciones en tweets no vistos durante el entrenamiento para evaluar cómo el modelo generaliza en datos nuevos.

### 4. **Visualización de Resultados**

Se incluyeron varias visualizaciones adicionales para interpretar los resultados:

- **Distribución de Sentimientos Predichos**: Un gráfico de barras que muestra la distribución de tweets clasificados como positivos o negativos.

## Conclusiones

El modelo entrenado logró una **exactitud de 0.8815**, lo que indica un buen desempeño en la clasificación de sentimientos. A lo largo del proceso, se realizaron varias modificaciones en la arquitectura del modelo y en los hiperparámetros, lo que mejoró su rendimiento. Las técnicas de regularización, como el **Dropout**, ayudaron a evitar el sobreajuste y a mejorar la generalización del modelo.

El análisis adicional mediante las visualizaciones de la distribución de sentimientos y las nubes de palabras proporcionaron información valiosa sobre cómo el modelo está clasificando los tweets y qué palabras son más representativas de cada sentimiento.

## Estructura del Proyecto

El proyecto incluye los siguientes archivos:

- **`sentiment_analysis.py`**: El archivo principal del proyecto. Contiene el código para limpiar los datos, construir y entrenar el modelo, evaluar el rendimiento y realizar predicciones sobre nuevos tweets.
- **`sentiment_analysis.ipynb`**: Un archivo Jupyter Notebook que proporciona una implementación más interactiva del análisis de sentimientos. Este notebook incluye la visualización de métricas, el análisis de resultados, y la predicción de sentimientos en nuevos tweets. Es útil para exploración interactiva y análisis más detallados del proceso.
- **`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto en un entorno virtual.

Por motivos de tamaño, no pude incluir el dataset, pero se encuentra en esta liga de kaggle: https://www.kaggle.com/datasets/kazanova/sentiment140

## Requisitos

- Python 3.x
- Keras
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- WordCloud
- Scikit-learn

## Instrucciones de Instalación

[De momento el archivo .py está roto; descargar unicamente el archivo .ipynb y usar ese mismo]

### 1. Crear un Entorno Virtual

Se recomienda crear un entorno virtual para gestionar las dependencias del proyecto. A continuación se detallan los pasos:

1. **Instalar `virtualenv` (si no lo tienes instalado)**:
   ```bash
   pip install virtualenv
   ```
2. **Crear el entorno virtual**:
    ```bash
    python3 -m venv myenv
   ```
3. **Activar el entorno virtual**:
- En Windows:
    ```bash
   .\venv\Scripts\activate

- En MacOS/Linux
    ```bash
   source venv/bin/activate
    
4. **Instalar las dependencias**  : Una vez dentro del entorno virtual, instala las dependencias del proyecto usando el archivo `requirements.txt`:
    ```bash
   pip install -r requirements.txt
   ```

### 2. Ejecutar el Proyecto

Una vez configurado el entorno virtual y las dependencias, puedes entrenar y evaluar el modelo:

- **Entrenar el modelo**:
  ```bash
  python train_model.py

- **Generar las visualizaciones y analizar los resultados**:
  ```bash
  python train_model.py
   ```

















