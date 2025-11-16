import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicción de Nivel de Estrés",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Carga de Componentes del Modelo ---
@st.cache_resource
def load_assets():
    """Carga el modelo, scaler, y encoders guardados."""
    try:
        # Rutas de los archivos guardados
        model_path = 'best_model.pkl'
        scaler_path = 'scaler.pkl'
        label_encoders_path = 'label_encoders.pkl'
        feature_names_path = 'feature_names.pkl'
        target_encoder_path = 'target_encoder.pkl'

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        with open(label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
            
        with open(target_encoder_path, 'rb') as f:
            target_encoder = pickle.load(f)

        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)

        return model, scaler, label_encoders, target_encoder, feature_names
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos: Uno o más archivos .pkl no se encuentran. Asegúrate de que el script de entrenamiento se haya ejecutado y haya guardado: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error al cargar los componentes del modelo: {e}")
        st.stop()

model, scaler, label_encoders, target_encoder, feature_names = load_assets()

# Definir la lista de ocupaciones (usando las del train.csv si está disponible)
# En un despliegue real, se leería de un archivo de configuración o la info_train
try:
    train_df = pd.read_csv('train.csv')
    occupation_options = sorted(train_df['Occupation'].unique().tolist())
    bmi_options = sorted(train_df['BMI_Category'].unique().tolist())
    sleep_disorder_options = sorted(train_df['Sleep_Disorder'].unique().tolist())
except Exception:
    # Opciones de fallback si 'train.csv' no está disponible o no se puede leer
    occupation_options = ['Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Manager', 'Nurse', 'Sales Representative', 'Scientist', 'Software Engineer', 'Teacher']
    bmi_options = ['Normal', 'Overweight', 'Obese']
    sleep_disorder_options = ['None', 'Sleep Apnea', 'Insomnia']


# --- Función de Predicción ---
def make_prediction(input_data):
    """
    Procesa los datos de entrada, los escala y predice el nivel de estrés.
    """
    # 1. Crear DataFrame con las features en el orden correcto
    # La lista 'feature_names' asegura el orden correcto de las columnas
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # 2. Codificar variables categóricas
    for col, encoder in label_encoders.items():
        # Usa el encoder para transformar el valor de entrada
        try:
            # Encuentra el índice de la categoría
            input_df[col] = encoder.transform(input_df[col])
        except ValueError:
             # Manejo de categorías no vistas (si es necesario)
             st.warning(f"Categoría no vista para {col}: {input_df[col].iloc[0]}")
             # Podríamos asignar un valor por defecto o la moda si es crítico, 
             # pero por simplicidad, asumimos que todas las categorías fueron vistas.
             pass

    # 3. Escalar features numéricas
    X_scaled = scaler.transform(input_df)
    
    # 4. Predicción
    prediction_encoded = model.predict(X_scaled)[0]
    prediction_proba = model.predict_proba(X_scaled)[0]
    
    # 5. Decodificar la predicción
    prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]
    
    # La probabilidad de la clase positiva (ESTRESADO, que es la clase '1')
    proba_positive = prediction_proba[target_encoder.transform(['ESTRESADO'])[0]]
    
    return prediction_label, proba_positive


# --- Interfaz de Usuario ---
st.title("🧠 Sistema de Clasificación de Nivel de Estrés")
st.markdown("Utiliza el **mejor modelo** entrenado para predecir si una persona tiene **Estrés Moderado** o está **Estresada** basándose en sus hábitos y métricas de salud.")
st.write("---")

# Estructura de la interfaz
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.header("👤 Datos Personales")
    age = st.slider("Edad (Age)", 18, 100, 35)
    occupation = st.selectbox("Ocupación (Occupation)", options=occupation_options)

with col2:
    st.header("😴 Hábitos de Sueño")
    sleep_duration = st.slider("Duración del Sueño (Sleep_Duration, horas)", 4.0, 10.0, 7.5, 0.1)
    quality_of_sleep = st.slider("Calidad del Sueño (Quality_of_Sleep, escala 1-10)", 1, 10, 7)
    sleep_disorder = st.selectbox("Trastorno del Sueño (Sleep_Disorder)", options=sleep_disorder_options)

with col3:
    st.header("🏃‍♀️ Métrica de Salud")
    physical_activity_level = st.slider("Nivel de Actividad Física (Physical_Activity_Level, mins/día)", 0, 150, 60)
    heart_rate = st.slider("Frecuencia Cardíaca (Heart_Rate, bpm)", 50, 100, 70)
    systolic_bp = st.slider("Presión Arterial Sistólica (Systolic_BP, mmHg)", 80, 200, 120)
    diastolic_bp = st.slider("Presión Arterial Diastólica (Diastolic_BP, mmHg)", 50, 150, 80)
    bmi_category = st.selectbox("Categoría de IMC (BMI_Category)", options=bmi_options)

st.write("---")

# --- Botón de Predicción y Resultados ---
if st.button("📈 Predecir Nivel de Estrés", type="primary"):
    
    # 1. Recopilar datos de entrada en formato de diccionario
    input_data = {
        'Age': age,
        'Sleep_Duration': sleep_duration,
        'Quality_of_Sleep': quality_of_sleep,
        'Physical_Activity_Level': physical_activity_level,
        'Heart_Rate': heart_rate,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp,
        'Occupation': occupation,
        'BMI_Category': bmi_category,
        'Sleep_Disorder': sleep_disorder
    }
    
    # Se eliminaron 'Person_ID' y 'Blood_Pressure' (que era el target),
    # y 'Blood_Pressure' (compuesto) que fue descartada en tu preprocesamiento

    # 2. Realizar la predicción
    prediction_label, proba_positive = make_prediction(input_data)
    
    # 3. Mostrar resultados
    st.header("Resultado de la Predicción")
    
    # Definir el estilo de la alerta
    if prediction_label == 'ESTRESADO':
        st.error(f"⚠️ **Clasificación: {prediction_label}**")
        message = "El modelo sugiere que los factores ingresados están fuertemente asociados con niveles altos de estrés. Se recomienda una revisión profesional."
        icon = "🚨"
    else:
        st.success(f"✅ **Clasificación: {prediction_label}**")
        message = "El modelo sugiere niveles de estrés moderados o normales. Se recomienda mantener o mejorar los hábitos de vida."
        icon = "👍"
    
    st.markdown(f"**{icon} {message}**")
    
    # Mostrar probabilidad
    st.metric(
        label="Probabilidad de estar Estresado (ESTRESADO)", 
        value=f"{proba_positive*100:.2f}%", 
        delta_color="off"
    )

    st.markdown("""
        <style>
        .stMetric > div:nth-child(1) {
            font-size: 1.2em;
            font-weight: bold;
        }
        .stMetric > div:nth-child(2) > div:nth-child(1) {
            font-size: 3em;
            color: #E94E77; /* Color para destacar la probabilidad */
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Mostrar el input en detalle
    st.subheader("Datos de Entrada Analizados")
    st.dataframe(pd.DataFrame([input_data]).T.rename(columns={0: "Valor Ingresado"}), use_container_width=True)

# --- Información Adicional ---
st.sidebar.title("Información del Proyecto")
st.sidebar.info(
    "Este modelo fue entrenado utilizando el algoritmo **" 
    f"{model.__class__.__name__}** " 
    "para predecir la categoría de estrés ('ESTRES_MODERADO' o 'ESTRESADO') "
    "basado en hábitos de sueño y métricas de salud."
)
st.sidebar.markdown(f"**Mejor Modelo Seleccionado:** `{model.__class__.__name__}`")
st.sidebar.markdown("**Métrica de Validación:** `F1-Score`")
st.sidebar.markdown("---")
# Aquí puedes agregar el link a tu página web/repositorio
st.sidebar.markdown("### Enlace al Proyecto")
st.sidebar.markdown("[Ir al Repositorio de Hábitos de Sueño (Link Ficticio)]("
                    "https://github.com/tu_usuario/tu_repo_sueno)")