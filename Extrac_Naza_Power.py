import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io # Necesario para la descarga de imágenes

# --- Estilo APA para Matplotlib ---
def aplicar_estilo_apa(fig, ax):
    """Aplica un estilo base APA en blanco y negro a una figura y ejes de Matplotlib."""
    # Colores y fuentes
    plt.style.use('grayscale') # Estilo base en escala de grises
    plt.rcParams['font.family'] = 'sans-serif' # Fuente sans-serif común en APA
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans'] # Opciones de fuente
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black' # Color del título del eje
    plt.rcParams['figure.facecolor'] = 'white' # Fondo de la figura
    plt.rcParams['axes.facecolor'] = 'white' # Fondo de los ejes

    # Títulos y etiquetas
    ax.title.set_fontsize(12) # Tamaño de fuente para títulos
    ax.xaxis.label.set_fontsize(10) # Tamaño de fuente para etiquetas de ejes
    ax.yaxis.label.set_fontsize(10)
    ax.tick_params(axis='both', which='major', labelsize=8) # Tamaño de fuente para ticks

    # Líneas y marcadores
    for line in ax.get_lines():
        line.set_linewidth(1.5) # Grosor de línea
        if line.get_marker() is not None and line.get_marker() != 'None':
             line.set_markersize(5) # Tamaño de marcador
             line.set_markeredgecolor('black')
             line.set_markerfacecolor('white') # Marcadores huecos si es posible

    # Grid (opcional, APA a veces prefiere sin grid o muy sutil)
    ax.grid(False) # Desactivar grid por defecto o usar:
    # ax.grid(True, linestyle=':', linewidth=0.5, color='lightgray', alpha=0.7)

    # Bordes de los ejes (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    # Leyenda (si existe)
    if ax.get_legend() is not None:
        legend = ax.get_legend()
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(0.5)
        for text in legend.get_texts():
            text.set_fontsize(8)
            text.set_color('black')

    fig.tight_layout() # Ajustar para evitar solapamientos

def consultar_nasa_power(longitud, latitud, fecha_inicio, fecha_fin, parametros):
    """
    Consulta la API de NASA POWER para obtener datos meteorológicos.
    """
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
            response.raise_for_status() # Lanza una excepción para errores HTTP
            data = response.json()

            properties = data.get("properties", {})
            parameter_data = properties.get("parameter", {})

            if not parameter_data:
                st.warning("No se encontraron datos para los parámetros y fechas seleccionadas.")
                return pd.DataFrame()

            df_data = {}
            # Intentar obtener las fechas de la estructura de geometría si está disponible
            # o del primer parámetro como fallback
            if "geometry" in data and "coordinates" in data["geometry"] and len(data["geometry"]["coordinates"]) > 2:
                 # Esta parte es una suposición, la estructura de fechas puede variar.
                 # Ajustar según la estructura real de la respuesta de la API si es necesario.
                 # dates = data["geometry"]["coordinates"][2].keys() # Ejemplo, puede no ser correcto
                 pass # Necesitaría ver la estructura exacta para las fechas aquí.

            # Fallback a obtener fechas del primer parámetro
            if not df_data or "Fecha" not in df_data:
                first_param_key = list(parameter_data.keys())[0]
                dates = list(parameter_data[first_param_key].keys())
                df_data["Fecha"] = [datetime.strptime(date_str, '%Y%m%d') for date_str in dates]


            for param_key, param_values in parameter_data.items():
                # Asegurarse de que los valores coincidan con el orden de las fechas
                # y manejar valores faltantes (-999 es común en NASA POWER)
                df_data[param_key] = [param_values.get(date_str, np.nan) if isinstance(param_values, dict) else np.nan for date_str in dates]
                # Reemplazar -999 con NaN si es necesario
                if param_key in df_data:
                     df_data[param_key] = [np.nan if x == -999 else x for x in df_data[param_key]]


            df = pd.DataFrame(df_data)
            if "Fecha" in df.columns:
                df.set_index("Fecha", inplace=True)
                df.sort_index(inplace=True)
            else:
                st.error("No se pudo construir la columna de Fechas correctamente.")
                return pd.DataFrame()

            return df
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Error HTTP al consultar NASA POWER: {http_err}")
        st.error(f"Respuesta del servidor: {response.text if 'response' in locals() else 'No response object'}")
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Error de conexión al consultar NASA POWER: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"Timeout al consultar NASA POWER: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Error general al consultar NASA POWER: {req_err}")
    except ValueError as val_err:
        st.error(f"Error al procesar los datos JSON de NASA POWER: {val_err}")
        st.error(f"Datos recibidos (parcial): {data if 'data' in locals() else 'No data object'}")
    except Exception as e:
        st.error(f"Un error inesperado ocurrió: {e}")
    return pd.DataFrame()


def generar_estadisticas(df):
    """
    Genera estadísticas descriptivas para un DataFrame.
    """
    if df.empty:
        st.warning("No hay datos para generar estadísticas.")
        return None # Devolver None si no hay datos

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

    var_descriptions = {
        "T2M": "Temperatura a 2 metros (°C)",
        "ALLSKY_SFC_PAR_TOT": "Radiación PAR incidente (MJ/m²/día)"
    }

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]): # Solo calcular para columnas numéricas
            stats_data["Variable"].append(var_descriptions.get(col, col))
            stats_data["Media"].append(round(df[col].mean(), 2))
            stats_data["Mediana"].append(round(df[col].median(), 2))
            stats_data["Suma"].append(round(df[col].sum(), 2))
            stats_data["Desv. Estándar"].append(round(df[col].std(), 2))
            stats_data["Mínimo"].append(round(df[col].min(), 2))
            stats_data["Máximo"].append(round(df[col].max(), 2))
            stats_data["Cuenta"].append(df[col].count())

    stats_df = pd.DataFrame(stats_data)
    return stats_df

# --- Configuración de la página de Streamlit ---
st.set_page_config(layout="wide", page_title="NASA POWER Data Explorer")
st.title("Interfaz de Consulta a NASA POWER")

# --- Panel lateral para entradas del usuario ---
with st.sidebar:
    st.header("Opciones de Consulta")
    # Configuración inicial de fechas
    fecha_fin_default = datetime.today()
    fecha_inicio_default = fecha_fin_default - timedelta(days=180) # Por defecto 180 días

    fecha_inicio_input = st.date_input("Fecha de inicio", value=fecha_inicio_default, help="Seleccione la fecha de inicio para la consulta.")
    fecha_fin_input = st.date_input("Fecha de fin", value=fecha_fin_default, help="Seleccione la fecha de fin para la consulta.")

    if fecha_inicio_input > fecha_fin_input:
        st.error("Error: La fecha de inicio no puede ser posterior a la fecha de fin.")
    else:
        # Convertir las fechas a string en el formato YYYY-MM-DD para la API
        fecha_inicio_str = fecha_inicio_input.strftime('%Y-%m-%d')
        fecha_fin_str = fecha_fin_input.strftime('%Y-%m-%d')


    col_lat, col_lon = st.columns(2)
    with col_lat:
        latitud = st.number_input("Latitud", value=-30.0, min_value=-90.0, max_value=90.0, format="%.4f", help="Latitud en grados decimales (ej. -30.0).")
    with col_lon:
        longitud = st.number_input("Longitud", value=-60.0, min_value=-180.0, max_value=180.0, format="%.4f", help="Longitud en grados decimales (ej. -60.0).")

    variables_disponibles = {
        "Temperatura a 2 metros (T2M)": "T2M",
        "Radiación PAR incidente (ALLSKY_SFC_PAR_TOT)": "ALLSKY_SFC_PAR_TOT"
        # Puedes añadir más variables aquí si es necesario
    }
    variables_seleccionadas_nombres = st.multiselect(
        "Seleccione las variables a consultar:",
        options=list(variables_disponibles.keys()),
        default=list(variables_disponibles.keys())[0:1], # Seleccionar la primera por defecto
        help="Elija una o más variables meteorológicas."
    )

    ejecutar_consulta = st.button("🚀 Ejecutar Consulta", type="primary", use_container_width=True)

# --- Estado de la sesión para almacenar resultados ---
if 'resultado_df' not in st.session_state:
    st.session_state.resultado_df = pd.DataFrame()
if 'fig_temporal' not in st.session_state:
    st.session_state.fig_temporal = None
if 'fig_histogramas' not in st.session_state:
    st.session_state.fig_histogramas = {}
if 'fig_boxplots' not in st.session_state:
    st.session_state.fig_boxplots = {}
if 'fig_mensual' not in st.session_state:
    st.session_state.fig_mensual = {}


# --- Lógica principal al ejecutar la consulta ---
if ejecutar_consulta:
    if not variables_seleccionadas_nombres:
        st.warning("⚠️ Por favor, seleccione al menos una variable para consultar.")
    elif fecha_inicio_input > fecha_fin_input:
        st.error("Error en fechas: La fecha de inicio debe ser anterior o igual a la fecha final.")
    else:
        parametros_api = [variables_disponibles[nombre] for nombre in variables_seleccionadas_nombres]
        df_resultado_consulta = consultar_nasa_power(longitud, latitud, fecha_inicio_str, fecha_fin_str, parametros_api)
        st.session_state.resultado_df = df_resultado_consulta
        # Limpiar figuras anteriores
        st.session_state.fig_temporal = None
        st.session_state.fig_histogramas = {}
        st.session_state.fig_boxplots = {}
        st.session_state.fig_mensual = {}


# --- Mostrar resultados si existen datos ---
if not st.session_state.resultado_df.empty:
    st.subheader("📊 Datos Obtenidos")
    df_display = st.session_state.resultado_df.copy()
    # Formatear índice de fecha para visualización
    try:
        df_display.index = df_display.index.strftime('%Y-%m-%d')
    except AttributeError:
        st.warning("El índice de fechas no pudo ser formateado. Verifique los datos.")


    rename_dict_display = {
        "T2M": "Temperatura (°C)",
        "ALLSKY_SFC_PAR_TOT": "Radiación PAR (MJ/m²/día)"
    }
    df_display_renamed = df_display.rename(columns=rename_dict_display)
    st.dataframe(df_display_renamed, use_container_width=True)

    # --- Gráfico temporal ---
    st.subheader("📈 Gráfico Temporal de Variables")
    fig_temporal, ax_temporal = plt.subplots(figsize=(10, 5)) # Ajustar tamaño para tesis
    aplicar_estilo_apa(fig_temporal, ax_temporal) # Aplicar estilo APA

    color_t2m = 'black'
    linestyle_t2m = '-'
    marker_t2m = 'o'

    color_par = 'dimgray' # Un gris oscuro diferente para la segunda variable
    linestyle_par = '--'
    marker_par = '^'


    # Eje primario (siempre existirá si hay datos)
    ax1_temporal = ax_temporal
    plotted_t2m = False
    plotted_par = False

    if "T2M" in st.session_state.resultado_df.columns:
        ax1_temporal.plot(st.session_state.resultado_df.index, st.session_state.resultado_df["T2M"],
                        color=color_t2m, linestyle=linestyle_t2m, marker=marker_t2m, markersize=3,
                        label="Temperatura (°C)")
        ax1_temporal.set_ylabel("Temperatura (°C)", color=color_t2m)
        ax1_temporal.tick_params(axis='y', labelcolor=color_t2m)
        plotted_t2m = True

    # Segundo eje Y si hay datos de PAR y también de T2M
    if "ALLSKY_SFC_PAR_TOT" in st.session_state.resultado_df.columns:
        if plotted_t2m: # Si ya se graficó T2M, usar eje secundario
            ax2_temporal = ax1_temporal.twinx()
            aplicar_estilo_apa(fig_temporal, ax2_temporal) # Aplicar estilo al segundo eje también
            ax2_temporal.plot(st.session_state.resultado_df.index, st.session_state.resultado_df["ALLSKY_SFC_PAR_TOT"],
                            color=color_par, linestyle=linestyle_par, marker=marker_par, markersize=3,
                            label="Radiación PAR (MJ/m²/día)")
            ax2_temporal.set_ylabel("Radiación PAR (MJ/m²/día)", color=color_par)
            ax2_temporal.tick_params(axis='y', labelcolor=color_par)
            # Ajustar spines para el segundo eje si es necesario
            ax2_temporal.spines['top'].set_visible(False)
            ax2_temporal.spines['left'].set_visible(False) # Ocultar el izquierdo si el primario ya lo tiene
            ax2_temporal.spines['bottom'].set_visible(False)
            ax2_temporal.spines['right'].set_linewidth(1)
            ax2_temporal.spines['right'].set_color('black')
            plotted_par = True
        else: # Si no hay T2M, graficar PAR en el eje primario
            ax1_temporal.plot(st.session_state.resultado_df.index, st.session_state.resultado_df["ALLSKY_SFC_PAR_TOT"],
                            color=color_par, linestyle=linestyle_par, marker=marker_par, markersize=3,
                            label="Radiación PAR (MJ/m²/día)")
            ax1_temporal.set_ylabel("Radiación PAR (MJ/m²/día)", color=color_par)
            ax1_temporal.tick_params(axis='y', labelcolor=color_par)
            plotted_par = True

    ax1_temporal.set_xlabel("Fecha")
    fig_temporal.suptitle("Datos Climáticos Diarios de NASA POWER", fontsize=14, fontweight='bold') # Título general APA
    #plt.title("Datos climáticos de NASA POWER") # Título original, ahora en suptitle

    # Configurar leyenda combinada APA style
    handles, labels = [], []
    if plotted_t2m:
        h1, l1 = ax1_temporal.get_legend_handles_labels()
        handles.extend(h1)
        labels.extend(l1)
    if plotted_par and 'ax2_temporal' in locals(): # Si se usó el segundo eje
        h2, l2 = ax2_temporal.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)
    elif plotted_par and not plotted_t2m: # Si PAR se graficó en el eje primario
        h1, l1 = ax1_temporal.get_legend_handles_labels() # Ya debería estar en handles/labels
        pass


    if handles: # Solo mostrar leyenda si hay algo que mostrar
        # Posicionar leyenda fuera del gráfico (estilo APA común)
        # ax1_temporal.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=False, shadow=False, ncol=2, frameon=True, edgecolor='black')
        # O leyenda dentro si es preferible y hay espacio:
        ax1_temporal.legend(handles, labels, loc='best', frameon=True, edgecolor='black', facecolor='white', framealpha=0.8)


    # Ajustar layout después de añadir todos los elementos
    fig_temporal.tight_layout(rect=[0, 0.05, 1, 0.95]) # rect=[left, bottom, right, top] para dejar espacio al suptitle y leyenda
    st.pyplot(fig_temporal)
    st.session_state.fig_temporal = fig_temporal # Guardar figura en session state

    # --- Opciones de descarga ---
    st.markdown("---")
    st.subheader("📥 Opciones de Descarga")
    col_desc_datos, col_desc_graf_temp, col_desc_stats = st.columns(3)

    with col_desc_datos:
        csv_data = st.session_state.resultado_df.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Datos (CSV)",
            data=csv_data,
            file_name=f"datos_nasa_power_{fecha_inicio_str}_{fecha_fin_str}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Descarga los datos tabulares en formato CSV."
        )

    with col_desc_graf_temp:
        if st.session_state.fig_temporal:
            img_temporal_buf = io.BytesIO()
            st.session_state.fig_temporal.savefig(img_temporal_buf, format="png", dpi=300, bbox_inches='tight') # dpi alto para calidad
            img_temporal_buf.seek(0)
            st.download_button(
                label="Descargar Gráfico Temporal (PNG)",
                data=img_temporal_buf,
                file_name=f"grafico_temporal_nasa_power_{fecha_inicio_str}_{fecha_fin_str}.png",
                mime="image/png",
                use_container_width=True,
                help="Descarga el gráfico temporal en formato PNG."
            )

    # --- Sección de estadísticas ---
    st.markdown("---")
    st.subheader("🧮 Estadísticas Descriptivas")
    stats_df = generar_estadisticas(st.session_state.resultado_df)

    if stats_df is not None and not stats_df.empty:
        st.dataframe(stats_df.set_index("Variable"), use_container_width=True) # Poner Variable como índice para mejor visualización
        with col_desc_stats:
            stats_csv = stats_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Estadísticas (CSV)",
                data=stats_csv,
                file_name=f"estadisticas_nasa_power_{fecha_inicio_str}_{fecha_fin_str}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Descarga las estadísticas descriptivas en formato CSV."
            )

        # --- Gráficos estadísticos (Histogramas y Boxplots) ---
        st.subheader("📊 Gráficos Estadísticos Adicionales")

        if len(st.session_state.resultado_df.columns) > 0:
            # Crear pestañas para cada variable numérica
            numeric_cols = [col for col in st.session_state.resultado_df.columns if pd.api.types.is_numeric_dtype(st.session_state.resultado_df[col])]
            if not numeric_cols:
                st.info("No hay variables numéricas para generar gráficos estadísticos.")

            tabs_graf_stats = st.tabs([rename_dict_display.get(col, col) for col in numeric_cols])

            for i, col_name in enumerate(numeric_cols):
                with tabs_graf_stats[i]:
                    st.markdown(f"#### Análisis para: {rename_dict_display.get(col_name, col_name)}")
                    col_g1, col_g2 = st.columns(2)

                    # Histograma
                    with col_g1:
                        st.markdown("**Histograma de Frecuencias**")
                        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                        aplicar_estilo_apa(fig_hist, ax_hist)
                        # Usar st.session_state.resultado_df[col_name] para los datos
                        ax_hist.hist(st.session_state.resultado_df[col_name].dropna(), bins=15, color='silver', edgecolor='black', alpha=0.9)
                        ax_hist.set_title(f'Distribución de {rename_dict_display.get(col_name, col_name)}', fontsize=10)
                        ax_hist.set_xlabel('Valor', fontsize=9)
                        ax_hist.set_ylabel('Frecuencia', fontsize=9)
                        fig_hist.tight_layout()
                        st.pyplot(fig_hist)
                        st.session_state.fig_histogramas[col_name] = fig_hist # Guardar

                        img_hist_buf = io.BytesIO()
                        fig_hist.savefig(img_hist_buf, format="png", dpi=300, bbox_inches='tight')
                        img_hist_buf.seek(0)
                        st.download_button(
                            label=f"Descargar Histograma (PNG)",
                            data=img_hist_buf,
                            file_name=f"histograma_{col_name}_{fecha_inicio_str}_{fecha_fin_str}.png",
                            mime="image/png",
                            key=f"dl_hist_{col_name}",
                            use_container_width=True
                        )

                    # Diagrama de Caja (Boxplot)
                    with col_g2:
                        st.markdown("**Diagrama de Caja y Bigotes**")
                        fig_box, ax_box = plt.subplots(figsize=(6, 4))
                        aplicar_estilo_apa(fig_box, ax_box)
                        ax_box.boxplot(st.session_state.resultado_df[col_name].dropna(), vert=False, patch_artist=True,
                                       boxprops=dict(facecolor='silver', color='black'),
                                       medianprops=dict(color='black', linewidth=1.5),
                                       whiskerprops=dict(color='black'),
                                       capprops=dict(color='black'),
                                       flierprops=dict(marker='.', markerfacecolor='black', markeredgecolor='none', markersize=5))
                        ax_box.set_title(f'Dispersión de {rename_dict_display.get(col_name, col_name)}', fontsize=10)
                        ax_box.set_yticklabels([]) # Sin etiquetas en el eje Y para boxplot horizontal
                        ax_box.set_xlabel('Valor', fontsize=9)
                        fig_box.tight_layout()
                        st.pyplot(fig_box)
                        st.session_state.fig_boxplots[col_name] = fig_box # Guardar

                        img_box_buf = io.BytesIO()
                        fig_box.savefig(img_box_buf, format="png", dpi=300, bbox_inches='tight')
                        img_box_buf.seek(0)
                        st.download_button(
                            label=f"Descargar Diagrama de Caja (PNG)",
                            data=img_box_buf,
                            file_name=f"boxplot_{col_name}_{fecha_inicio_str}_{fecha_fin_str}.png",
                            mime="image/png",
                            key=f"dl_box_{col_name}",
                            use_container_width=True
                        )

                    # Estadísticas detalladas por variable
                    st.write(f"**Estadísticas Detalladas para {rename_dict_display.get(col_name, col_name)}:**")
                    desc_df = st.session_state.resultado_df[col_name].describe().to_frame().T
                    st.dataframe(desc_df, use_container_width=True)

                    # Gráfico de Promedios Mensuales
                    if len(st.session_state.resultado_df.index.month.unique()) > 1 and len(st.session_state.resultado_df) > 28 : # Solo si hay más de un mes
                        st.markdown(f"**Promedios Mensuales para {rename_dict_display.get(col_name, col_name)}**")
                        temp_df_monthly = st.session_state.resultado_df.copy()
                        # Asegurarse que el índice es DatetimeIndex
                        if not isinstance(temp_df_monthly.index, pd.DatetimeIndex):
                            try:
                                temp_df_monthly.index = pd.to_datetime(temp_df_monthly.index)
                            except Exception as e:
                                st.warning(f"No se pudo convertir el índice a Datetime para promedios mensuales: {e}")
                                continue # Saltar esta sección si la conversión falla

                        monthly_avg = temp_df_monthly[col_name].resample('M').mean() # 'M' para fin de mes, 'MS' para inicio
                        if not monthly_avg.empty:
                            fig_month, ax_month = plt.subplots(figsize=(8, 4))
                            aplicar_estilo_apa(fig_month, ax_month)
                            ax_month.plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-', color='black', markersize=4)
                            # ax_month.bar(monthly_avg.index, monthly_avg.values, color='silver', edgecolor='black', width=20) # Alternativa con barras
                            ax_month.set_title(f'Promedio Mensual de {rename_dict_display.get(col_name, col_name)}', fontsize=10)
                            ax_month.set_xlabel('Mes', fontsize=9)
                            ax_month.set_ylabel('Valor Promedio', fontsize=9)
                            plt.xticks(rotation=45, ha='right')
                            fig_month.tight_layout()
                            st.pyplot(fig_month)
                            st.session_state.fig_mensual[col_name] = fig_month # Guardar

                            img_month_buf = io.BytesIO()
                            fig_month.savefig(img_month_buf, format="png", dpi=300, bbox_inches='tight')
                            img_month_buf.seek(0)
                            st.download_button(
                                label=f"Descargar Promedios Mensuales (PNG)",
                                data=img_month_buf,
                                file_name=f"promedios_mensuales_{col_name}_{fecha_inicio_str}_{fecha_fin_str}.png",
                                mime="image/png",
                                key=f"dl_month_{col_name}",
                                use_container_width=True
                            )
                        else:
                            st.info("No hay suficientes datos para calcular promedios mensuales.")
    else:
        if ejecutar_consulta: # Solo mostrar si se intentó una consulta
             st.info("ℹ️ No se obtuvieron datos para la consulta realizada o los datos no son numéricos. Por favor, verifique los parámetros e intente nuevamente.")

# --- Pie de página o información adicional ---
st.markdown("---")
st.markdown("Desarrollado con Streamlit y Matplotlib. Datos de [NASA POWER](https://power.larc.nasa.gov/).")
