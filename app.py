
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(layout="wide")

st.title('Análisis del Impacto de la Inteligencia Artificial en Estudiantes')
st.write('Esta aplicación analiza cómo el uso de la IA generativa afecta el rendimiento académico y el riesgo de agotamiento en estudiantes.')

# Definir la ruta del archivo CSV
# Asumimos que el archivo CSV estará en una subcarpeta 'data' en el mismo directorio que app.py
csv_file_path = os.path.join('data', 'ai_student_impact_dataset.csv')

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: El archivo CSV no se encontró en la ruta: {path}. Por favor, asegúrate de que el archivo esté en la ubicación correcta.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el archivo CSV: {e}")
        return None

df_csv = load_data(csv_file_path)

if df_csv is not None:
    st.header('1. Vista Previa de los Datos')
    st.dataframe(df_csv.head())

    st.header('2. Estadísticas Descriptivas')
    st.dataframe(df_csv.describe(include='all'))

    # Limpieza de valores nulos (as per notebook, no nulls found, but good practice)
    df_csv_cleaned = df_csv.dropna()
    if df_csv.shape[0] != df_csv_cleaned.shape[0]:
        st.warning(f"Se eliminaron {df_csv.shape[0] - df_csv_cleaned.shape[0]} filas con valores nulos.")

    st.header('3. Visualizaciones')

    # Histograma de Post_Semester_GPA
    st.subheader('Distribución del Post_Semester_GPA')
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df_csv_cleaned['Post_Semester_GPA'], bins=20, kde=True, ax=ax1)
    ax1.set_title('Distribución del Post_Semester_GPA')
    ax1.set_xlabel('Post_Semester_GPA')
    ax1.set_ylabel('Frecuencia')
    ax1.grid(axis='y', alpha=0.75)
    st.pyplot(fig1)

    # Scatter plot Weekly_GenAI_Hours vs Post_Semester_GPA
    st.subheader('Relación entre Horas de Uso de IA y GPA Post-Semestre')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Weekly_GenAI_Hours', y='Post_Semester_GPA', data=df_csv_cleaned, alpha=0.6, ax=ax2)
    ax2.set_title('Relación entre Horas de Uso de IA y GPA Post-Semestre')
    ax2.set_xlabel('Horas Semanales de IA Generativa')
    ax2.set_ylabel('Post_Semester_GPA')
    ax2.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig2)

    # Box plot Institutional_Policy vs Post_Semester_GPA
    st.subheader('Impacto de Institutional_Policy en Post_Semester_GPA')
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='Institutional_Policy', y='Post_Semester_GPA', data=df_csv_cleaned, palette='viridis', ax=ax3)
    ax3.set_title('Post_Semester_GPA por Política Institucional')
    ax3.set_xlabel('Política Institucional')
    ax3.set_ylabel('Post_Semester_GPA')
    ax3.grid(axis='y', alpha=0.75)
    st.pyplot(fig3)

    st.header('4. Preparación de Datos y Entrenamiento del Modelo')

    # Definir las características (X) y la variable objetivo (y)
    X = df_csv_cleaned.drop('Burnout_Risk_Level', axis=1)
    y = df_csv_cleaned['Burnout_Risk_Level']

    # Excluir 'Student_ID' de las características
    X = X.drop('Student_ID', axis=1, errors='ignore')

    # Identificar columnas categóricas y numéricas
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Crear un preprocesador para aplicar One-Hot Encoding a las columnas categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.write(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
    st.write(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
    st.write(f"Columnas categóricas a codificar: {list(categorical_features)}")
    st.write(f"Columnas numéricas: {list(numerical_features)}")

    # Crear y entrenar el modelo RandomForestClassifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    model.fit(X_train, y_train)
    st.success("Modelo RandomForestClassifier entrenado exitosamente.")

    st.header('5. Evaluación del Modelo')

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Matriz de Confusión
    st.subheader('Matriz de Confusión')
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax_cm)
    ax_cm.set_xlabel('Predicción')
    ax_cm.set_ylabel('Valor Real')
    ax_cm.set_title('Matriz de Confusión para Burnout_Risk_Level')
    st.pyplot(fig_cm)

    # Reporte de Clasificación
    st.subheader('Reporte de Clasificación')
    class_report_str = classification_report(y_test, y_pred)
    st.text(class_report_str)

    st.subheader('Interpretación de los Resultados')
    st.write("El reporte de clasificación y la matriz de confusión muestran el rendimiento del modelo en la predicción del nivel de riesgo de agotamiento. Se puede observar la precisión, recall y F1-score para cada categoría (High, Low, Medium). La matriz de confusión detalla dónde el modelo acierta y dónde confunde las clases.")
