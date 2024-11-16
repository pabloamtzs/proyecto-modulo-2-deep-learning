# sentiment_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Función para limpiar los tweets
def clean_tweet(tweet):
    """
    Limpia el tweet eliminando URLs, menciones, hashtags, caracteres especiales y convirtiendo todo a minúsculas.
    """
    # Implementa aquí la limpieza del texto
    cleaned_tweet = tweet.lower()  # Ejemplo de limpieza: convertir a minúsculas
    return cleaned_tweet

# Función de preprocesamiento de datos
def preprocess_data(df):
    """
    Preprocesa los datos del dataframe: limpieza de texto, tokenización y padding.
    """
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(df['cleaned_text'])
    X = tokenizer.texts_to_sequences(df['cleaned_text'])
    X = pad_sequences(X, maxlen=100)
    y = df['label']  # Asumiendo que la columna 'label' tiene las etiquetas de sentimiento
    return X, y, tokenizer

# Función para construir el modelo LSTM
def build_model(input_shape):
    """
    Construye el modelo LSTM para la clasificación de sentimientos.
    """
    model = Sequential()
    model.add(Embedding(input_dim=20000, output_dim=128, input_length=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Función para entrenar el modelo
def train_model(X_train, y_train, X_val, y_val):
    """
    Entrena el modelo LSTM con los datos proporcionados.
    """
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)
    model.save('sentiment_model.h5')
    return model, history

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_test):
    """
    Evalúa el rendimiento del modelo en el conjunto de prueba.
    """
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convertimos las probabilidades a etiquetas binarias

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Función para graficar las métricas del entrenamiento
def plot_metrics(history):
    """
    Grafica la precisión y la pérdida del entrenamiento.
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

# Función para realizar predicciones en tweets nuevos
def predict_sentiment(model, tokenizer, tweet):
    """
    Predice el sentimiento de un tweet limpio utilizando el modelo entrenado.
    """
    tweet_cleaned = clean_tweet(tweet)
    tweet_sequence = tokenizer.texts_to_sequences([tweet_cleaned])
    tweet_padded = pad_sequences(tweet_sequence, maxlen=100)
    prediction = model.predict(tweet_padded)
    return "Positivo" if prediction > 0.5 else "Negativo"

# Función principal que orquesta todo
def main():
    # Cargar el dataset
    df = pd.read_csv('sentiment_dataset.csv')  # Ajusta el nombre del archivo según sea necesario
    
    # Preprocesar los datos
    X, y, tokenizer = preprocess_data(df)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model, history = train_model(X_train, y_train, X_test, y_test)

    # Evaluar el modelo
    evaluate_model(model, X_test, y_test)

    # Graficar métricas
    plot_metrics(history)

    # Realizar una predicción de ejemplo
    tweet_example = "I love this product! It's amazing."
    sentiment = predict_sentiment(model, tokenizer, tweet_example)
    print(f"Sentimiento del tweet: {sentiment}")

if __name__ == "__main__":
    main()
