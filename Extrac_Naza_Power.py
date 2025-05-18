
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def consultar_nasa_power(longitud, latitud, fecha_inicio, fecha_fin, parametros):
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    params = {
        "parameters": ','.join(parametros),
        "community": "AG",
        "longitude": longitud,
        "latitude": latitud,
        "start": fecha_inicio.replace('-', ''),
        "end": fecha_fin.replace('-', ''),
        "format": "JSON"
    }
    
    try:
        with st.spinner('Consultando datos de NASA POWER...'):
            response = requests.get(base_url, params=params, verify=True, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Procesamiento de datos para crear un DataFrame estructurado
            properties = data.get("properties", {})
            parameter_data = properties.get("parameter", {})
            
            # Si no hay datos, retornar DataFrame vacío
            if not parameter_data:
                return pd.DataFrame()
            
            # Estructura para almacenar datos
            df_data = {}
            dates = []
            
            # Obtenemos todas las fechas disponibles del primer parámetro
            first_param = list(parameter_data.keys())[0]
            for date_str in parameter_data[first_param].keys():
                dates.append(date_str)
            
            # Creamos un diccionario con todas las fechas y parámetros
            df_data["Fecha"] = dates
            
            # Agregamos cada parámetro consultado
            for param in parametros:
                if param in parameter_data:
                    df_data[param] = [parameter_data[param].get(date, np.nan) for date in dates]
            
            # Creamos el DataFrame y convertimos la fecha
            df = pd.DataFrame(df_data)
            df["Fecha"] = pd.to_datetime(df["Fecha"])
            df.set_index("Fecha", inplace=True)
            df.sort_index(inplace=True)
            
            return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error al consultar NASA POWER: {e}")
        return pd.DataFrame()

# Función para generar estadísticas
def generar_estadisticas(df):
    if df.empty:
        st.warning("No hay datos para generar estadísticas.")
        return
    
    # Crear diccionario para almacenar estadísticas y descripciones
    stats_data = {
        "Variable": [],
        "Media": [],
        "Mediana": [],
        "Suma": [],
        "Desv. Estándar": [],
        "Mínimo": [],
        "Máximo": [],
        "Cuenta": []
    }
    
    # Variables y sus descripciones
    var_descriptions = {
        "T2M": "Temperatura a 2 metros (°C)",
        "ALLSKY_SFC_PAR_TOT": "Radiación PAR incidente (MJ/m²/día)"
    }
    
    # Calcular estadísticas para cada columna
    for col in df.columns:
        stats_data["Variable"].append(var_descriptions.get(col, col))
        stats_data["Media"].append(round(df[col].mean(), 2))
        stats_data["Mediana"].append(round(df[col].median(), 2))
        stats_data["Suma"].append(round(df[col].sum(), 2))
        stats_data["Desv. Estándar"].append(round(df[col].std(), 2))
        stats_data["Mínimo"].append(round(df[col].min(), 2))
        stats_data["Máximo"].append(round(df[col].max(), 2))
        stats_data["Cuenta"].append(df[col].count())
    
    # Crear DataFrame de estadísticas
    stats_df = pd.DataFrame(stats_data)
    
    return stats_df

# Configuración inicial de fechas
fecha_fin_default = datetime.today()
fecha_inicio_default = fecha_fin_default - timedelta(days=180)

# Interfaz gráfica con Streamlit
st.set_page_config(layout="wide", page_title="NASA POWER Data Explorer")
st.title("Interfaz NASA POWER")

# Panel contraíble para inputs
with st.sidebar:
    st.header("Opciones de consulta")
    
    fecha_inicio = st.date_input("Fecha inicio", value=fecha_inicio_default)
    fecha_fin = st.date_input("Fecha fin", value=fecha_fin_default)
    
    # Validación de fechas
    if fecha_inicio > fecha_fin:
        st.error("Error: La fecha de inicio debe ser anterior a la fecha final")
    
    col1, col2 = st.columns(2)
    with col1:
        latitud = st.number_input("Latitud", value=-30.0, min_value=-90.0, max_value=90.0, format="%.4f")
    with col2:
        longitud = st.number_input("Longitud", value=-60.0, min_value=-180.0, max_value=180.0, format="%.4f")
    
    variables_disponibles = {
        "Temperatura a 2 metros (T2M)": "T2M",
        "Radiación PAR incidente (ALLSKY_SFC_PAR_TOT)": "ALLSKY_SFC_PAR_TOT"
    }
    
    variables_seleccionadas = st.multiselect(
        "Selecciona variables:",
        options=list(variables_disponibles.keys()),
        default=list(variables_disponibles.keys())
    )
    
    ejecutar_consulta = st.button("Ejecutar consulta", type="primary")

# Creamos un contenedor para almacenar los resultados entre ejecuciones
if 'resultado_df' not in st.session_state:
    st.session_state.resultado_df = pd.DataFrame()

# Panel principal para resultados
if ejecutar_consulta:
    if not variables_seleccionadas:
        st.warning("Debes seleccionar al menos una variable.")
    else:
        parametros = [variables_disponibles[var] for var in variables_seleccionadas]
        df_resultado = consultar_nasa_power(longitud, latitud, str(fecha_inicio), str(fecha_fin), parametros)
        
        # Guardar resultado en el estado de la sesión
        st.session_state.resultado_df = df_resultado

# Mostrar resultados si existen datos
if not st.session_state.resultado_df.empty:
    st.subheader("Datos obtenidos:")
    
    # Formatear datos para mostrar
    df_display = st.session_state.resultado_df.copy()
    df_display.index = df_display.index.strftime('%Y-%m-%d')
    
    # Renombrar columnas para mejor visualización
    rename_dict = {
        "T2M": "Temperatura (°C)",
        "ALLSKY_SFC_PAR_TOT": "Radiación PAR (MJ/m²/día)"
    }
    df_display = df_display.rename(columns=rename_dict)
    
    # Mostrar tabla
    st.dataframe(df_display, use_container_width=True)
    
    # Gráficos
    st.subheader("Gráfico temporal")
    
    fig = plt.figure(figsize=(12, 6))
    
    # Determinar si necesitamos uno o dos ejes Y
    ax1 = fig.add_subplot(111)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    if "T2M" in st.session_state.resultado_df.columns:
        ax1.plot(st.session_state.resultado_df.index, st.session_state.resultado_df["T2M"], 
                 'b-', marker='o', markersize=3, label="Temperatura (°C)")
        ax1.set_ylabel("Temperatura (°C)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
    
    # Segundo eje Y si hay datos de PAR
    if "ALLSKY_SFC_PAR_TOT" in st.session_state.resultado_df.columns:
        if "T2M" in st.session_state.resultado_df.columns:
            ax2 = ax1.twinx()
            ax2.plot(st.session_state.resultado_df.index, st.session_state.resultado_df["ALLSKY_SFC_PAR_TOT"], 
                     'r-', marker='s', markersize=3, label="Radiación PAR (MJ/m²/día)")
            ax2.set_ylabel("Radiación PAR (MJ/m²/día)", color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        else:
            ax1.plot(st.session_state.resultado_df.index, st.session_state.resultado_df["ALLSKY_SFC_PAR_TOT"], 
                     'r-', marker='s', markersize=3, label="Radiación PAR (MJ/m²/día)")
            ax1.set_ylabel("Radiación PAR (MJ/m²/día)", color='r')
    
    plt.title("Datos climáticos de NASA POWER")
    plt.xlabel("Fecha")
    
    # Configurar leyenda combinada
    handles = []
    labels = []
    
    if "T2M" in st.session_state.resultado_df.columns:
        h1, l1 = ax1.get_legend_handles_labels()
        handles.extend(h1)
        labels.extend(l1)
    
    if "ALLSKY_SFC_PAR_TOT" in st.session_state.resultado_df.columns and "T2M" in st.session_state.resultado_df.columns:
        h2, l2 = ax2.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)
    
    if handles:
        plt.legend(handles, labels, loc='upper left')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Opciones de descarga
    col1, col2 = st.columns(2)
    
    with col1:
        csv = st.session_state.resultado_df.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos CSV",
            data=csv,
            file_name=f"datos_nasa_power_{fecha_inicio}_{fecha_fin}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Sección de estadísticas mejorada
    st.subheader("Estadísticas básicas")
    
    # Mostrar estadísticas
    stats_df = generar_estadisticas(st.session_state.resultado_df)
    
    if stats_df is not None:
        st.dataframe(stats_df, use_container_width=True)
        
        # Botón para descargar estadísticas
        with col2:
            stats_csv = stats_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar estadísticas CSV",
                data=stats_csv,
                file_name=f"estadisticas_nasa_power_{fecha_inicio}_{fecha_fin}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Visualización de estadísticas
        st.subheader("Gráficos estadísticos")
        
        if len(st.session_state.resultado_df.columns) > 0:
            tabs = st.tabs([rename_dict.get(col, col) for col in st.session_state.resultado_df.columns])
            
            for i, col in enumerate(st.session_state.resultado_df.columns):
                with tabs[i]:
                    col1, col2 = st.columns(2)
                    
                    # Histograma
                    with col1:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(st.session_state.resultado_df[col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_title(f'Histograma - {rename_dict.get(col, col)}')
                        ax.set_xlabel('Valor')
                        ax.set_ylabel('Frecuencia')
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                    
                    # Boxplot
                    with col2:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.boxplot(st.session_state.resultado_df[col].dropna(), vert=False)
                        ax.set_title(f'Diagrama de caja - {rename_dict.get(col, col)}')
                        ax.set_ylabel('')
                        ax.set_xlabel('Valor')
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                    
                    # Estadísticas detalladas
                    st.write("#### Estadísticas detalladas")
                    
                    # Descripción de estadísticas
                    desc = st.session_state.resultado_df[col].describe().to_frame().T
                    st.dataframe(desc, use_container_width=True)
                    
                    # Valores mensuales (si hay suficientes datos)
                    if len(st.session_state.resultado_df) > 28:
                        st.write("#### Promedios mensuales")
                        
                        # Agregar columnas de mes y año
                        temp_df = st.session_state.resultado_df.copy()
                        temp_df['month'] = temp_df.index.month
                        temp_df['year'] = temp_df.index.year
                        
                        # Agrupar por mes
                        monthly_avg = temp_df.groupby('month')[col].mean().reset_index()
                        monthly_avg['month'] = monthly_avg['month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
                        
                        # Mostrar promedios mensuales
                        monthly_fig, monthly_ax = plt.subplots(figsize=(10, 5))
                        monthly_ax.bar(monthly_avg['month'], monthly_avg[col], color='skyblue')
                        monthly_ax.set_title(f'Promedio mensual - {rename_dict.get(col, col)}')
                        monthly_ax.set_xlabel('Mes')
                        monthly_ax.set_ylabel('Valor promedio')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(monthly_fig)

# Mostrar mensaje si no hay datos
if ejecutar_consulta and st.session_state.resultado_df.empty:
    st.warning("No se obtuvieron datos para la consulta realizada. Por favor verifica los parámetros e intenta nuevamente.")
