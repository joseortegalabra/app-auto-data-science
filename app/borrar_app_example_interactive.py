import streamlit as st
from streamlit_autorefresh import st_autorefresh
from google.cloud import bigquery

import os
from dotenv import load_dotenv

import pytz
from datetime import datetime

import pandas as pd
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly


################################# READ ENV VARIABLES - CREDENTIALS TO CONNECT TO BQ - saved in .env file #################################
load_dotenv()
env_var_project_gcp = os.environ.get("PROJECT_GCP", "")
env_var_path_sa = os.environ.get("PATH_SA", "")

PROJECT_ID = env_var_project_gcp
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = env_var_path_sa




################################# FUNTION SHOW ACTUAL TIME #################################

# actual time in chile timezone continental
def get_actual_time():
    """
    Get actual time chile in time zone Chile and format: "%H:%M:%S %d/%m/%Y"
    """
    chile_timezone = pytz.timezone('Chile/Continental')
    now = datetime.now(chile_timezone)
    format_time_chile = now.strftime("%H:%M:%S  %d/%m/%Y")
    return format_time_chile


################################# FUNCTION TO QUERY A DATAFRAME #################################
def GCPQuery2BigQuery(Project,InputQuery):
    """
    Execute query (string) given as input and return the output as a pandas dataframe

    Args:
        Project (string): project GCP
        InputQuery (string): query SQL string - bigquery

    Returns:
        df_results (dataframe): dataframe output of the query
    """
    client = bigquery.Client(project = Project)
    query_job = client.query(InputQuery)
    results = query_job.result()
    df_results = results.to_dataframe()
    return df_results



################################# FUNCTION TO GET COLOR OF A STRING ALERT #################################
def get_color_alerta_prediction(name_alert_prediction):
    """
    Given a string color of alert (low, medium, high // bajo, medio, alto) get a color to show in the app
    Map of colors:
    - bajo: green
    - medio: orange
    - alto: red

    Args:
        alert_prediction (string): name of the alert
    
    Returns:
        (string): color of the alert
    """
    if name_alert_prediction == "bajo":
        return "green"
    if name_alert_prediction == "medio":
        return "orange"
    if name_alert_prediction == "alto":
        return "red"


################################# XXXXX #################################
def identify_actual_alert_to_traffic_light(nivel_alert):
    """
    Auxiliar function feature 1.2 - traffic light actual level of alert. Identity the actual level of alert and generate an output for the traffic light plotly plot

    Output traffic light alerts
        colors = ['rgb(50, 50, 50)', 'orange', 'rgb(50, 50, 50)']
        sizes = [50, 75, 50]
    
    When it is turn off (no alert), the color take the values "rgb(50, 50, 50)" and the size of circle is "50"

    When it is turn on (alert), it take the values "green" and the size of circle is "75"
    """

    print('Actual level of alert: ', nivel_alert)

    # generar colors y size por defecto todo apagado. ORDEN: ['red', 'orange', 'green']
    colors = ['rgb(50, 50, 50)', 'rgb(50, 50, 50)', 'rgb(50, 50, 50)']
    sizes = [50, 50, 50]

    # encender (color y tamaño) del nivel de alerta actual
    if nivel_alert == "green":
        colors[2] = "green"
        sizes[2] = 75

    if nivel_alert == "orange":
        colors[1] = "orange"
        sizes[1] = 75

    if nivel_alert == "red":
        colors[0] = "red"
        sizes[0] = 75
        
    return colors, sizes

def make_plot_traffic_light_alerts(nivel_alert):
    """
    Given the actual level of alert, make a traffic light with turn on the level of alert
    """
    #### get colors and size of the traffic light - this configuration change according to the color alert "green", "orange", "red"
    colors, sizes = identify_actual_alert_to_traffic_light(nivel_alert)

    # PARAMS
    x_coords = [0.5, 0.5, 0.5]
    y_coords = [0.8, 0.5, 0.2]
    values = [1, 2, 3]
    labels = ['Red', 'Yellow', 'Green']

    #Do plot
    texts = [f"{label}<br>Value: {value}" for label, value in zip(labels, values)]
    hoverinfos = ['text', 'text', 'text']

    fig = go.Figure()

    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=[x_coords[i]],
                y=[y_coords[i]],
                text=texts[i],
                hoverinfo=hoverinfos[i],
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=sizes[i],
                    line=dict(
                        color='black',
                        width=2
                    )
                )
            )
        )

    fig.update_layout(
        xaxis=dict(
            range=[0, 1],
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[0, 1],
            showticklabels=False,
            zeroline=False
        ),
        #height=600,
        width=150,
        title = "Actual Alert",
        showlegend=False,
        title_x=0.5
    )
    return fig



################################# XXXXX #################################
def trend_true_vs_predicted(df):
    """
    Plot trend true values vs predicted values
    """
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
     
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Value_prediction'], mode='lines', name='Valor predicho', line=dict(color='blue', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Value_true'], mode='lines', name='Valor real', line=dict(color='grey', width=3)), row=1, col=1)

    fig.update_layout(
        title='Trend Target True vs Target Predicted (Last 8 hours)',
        xaxis_title='Date and Time',
        yaxis_title='Value Target',
        yaxis_title_standoff=30,
        #height=600, 
        #width=1000,
       # title_x=0.5
    )
    return fig



################################# plot trend barplots alerts #################################
def trend_barplot_alerts(df):
    """
    Make a barplot to represent the trend of the differents predictions of alerts: low, medium, high
    """

    ###### definir params ######

    # parámetros mapeo colores (bajo, medio, alto) (green, orange, red), (1,2,3)
    colors = {'bajo': 'green', 'medio': 'orange', 'alto': 'red'}
    number = {'bajo': 1, 'medio': 2, 'alto': 3}


    # obtener los valores únicos de las predicciones (por ejemplo que solo hallan prediciones bajo y no medio ni alto)
    alertas_unicas_generadas = df['Alert_prediction'].unique()

    # filtrar los parámetros de mapeo de acuerdo a los valors únicos predichos
    # por ejemplo si en las predicciones solo hay valores "bajo" y "alto", el valor "medio debe de borrarse de los params"
    colors = {clave: valor for clave, valor in colors.items() if clave in alertas_unicas_generadas}
    number = {clave: valor for clave, valor in number.items() if clave in alertas_unicas_generadas}
    cat_order = list(colors.keys())


    ###### transformaciones a los datos - de acuerdo a los params ######

    # Convertir la columna "Alert_prediction" en tipo categórico con un orden específico
    df['Alert_prediction'] = pd.Categorical(df['Alert_prediction'], categories=cat_order, ordered=True)

    ## Crear la columna "Bar_size" con el tamaño de las barras
    df['Bar_size'] = df['Alert_prediction'].map(number)



    ########################################################################################################################
    # Crear el gráfico de barras
    fig = px.bar(df, x='datetime', y='Bar_size', color='Alert_prediction', barmode='stack', color_discrete_map=colors)

    # Establecer el orden deseado en el eje y
    fig.update_layout(title='Trend Alerts Generated (Last 8 Hours)',
                      xaxis_title='Date and Time',
                      yaxis_title='Value Alert',
                      yaxis=dict(categoryorder='array', categoryarray=cat_order, tickvals=list(range(1, len(cat_order)+1)), ticktext=cat_order),
                      #title_x=0.5
                     )

    return fig





if __name__ == "__main__":

    ################################# set page configuration #################################
    st.set_page_config(layout="wide")
    
    
    ################################# update every 5 minutes interval = (minutes) * (seconds) * (1000) #################################
    # st_autorefresh(interval = 5 * 60 * 1000, key="dataframerefresh") # if it requiered more frecuency could generate an observer and if this value change, refresh all page
    count_refresh_page = st_autorefresh(interval = 5 * 60 * 1000)
    # try no key:  An optional key that uniquely identifies this component. If this is None, and the component's arguments are changed, the component will be re-mounted in the Streamlit frontend and lose its current state.
    
    
    ### Tittle page
    st.subheader("Example dashboard app that show the result of a ML model that is running in other cloud services")


    # show actual time
    time_now = get_actual_time()
    st.write(f"Actual timezone: {time_now}")


    ########### FEATURE 1 - INFO PREDICTION ACTUAL ###########

    # query table instance_predictions
    query_instance_prediction = f"""SELECT * FROM `{PROJECT_ID}.template_research_and_development_mvp_stacking.front_instance_predictions`"""
    table_instance_predictions = GCPQuery2BigQuery(f'{PROJECT_ID}', query_instance_prediction)

    # feature 1.1 info prediction actual
    FECHA_ULTIMA_ACTUALIZACION = table_instance_predictions['datetime'][0].strftime('%Y-%m-%d %H:%M:%S')
    FECHA_PREDICCION = table_instance_predictions['datetime_forecast'][0].strftime('%Y-%m-%d %H:%M:%S')
    VALOR_PREDICCION = np.round(table_instance_predictions['Value_prediction'].values[0], 2)
    ALERTA_PREDICCION = table_instance_predictions['Alert_prediction'].values[0]
    COLOR_ALERTA_PREDICCION = get_color_alerta_prediction(ALERTA_PREDICCION)

    # feature 1.2 traffic light -> alert actual prediction
    FIG_TRAFFIC_LIGHT = make_plot_traffic_light_alerts(nivel_alert = COLOR_ALERTA_PREDICCION)


    ########### FEATURE 2 - TRENDS ###########

    # query tables tendency_predictions
    query_tendency_predictions = f"""
    SELECT * 
    FROM f`{PROJECT_ID}.template_research_and_development_mvp_stacking.front_tendency_predictions`
    order by datetime asc
    """
    table_tendency_predictions = GCPQuery2BigQuery(f'{PROJECT_ID}', query_tendency_predictions)


    # feature 2.1 graficar tendencia numérica real vs predicho
    FIG_TREND_TRUE_VS_PRED = trend_true_vs_predicted(df = table_tendency_predictions)

    
    # feature 2.2 graficar tendencia alertas generadas
    FIG_TREND_BARPLOT_ALERTS = trend_barplot_alerts(df = table_tendency_predictions)
    


    ########### SHOW VALUES, PLOTS IN UI STREAMLIT ###########

    ###### Firts group of components, alert
    col1_alert, col2_alert, col3_alert, col4_alert = st.columns(4)
    
    # "metrics" - "show values of the prediction"
    col1_alert.metric(label = "Datetime last update webpage", value = FECHA_ULTIMA_ACTUALIZACION)
    col2_alert.metric(label = "Datetime Prediction Model", value = FECHA_PREDICCION)
    col3_alert.metric(label = "Value Predicted", value = VALOR_PREDICCION)
    col4_alert.metric(label = "Alert Level Prediction", value = ALERTA_PREDICCION)


    ###### Second group of components, trend of alerts
    col1_trend, col2_trend = st.columns(2)

    # figure trend true vs pred
    col1_trend.plotly_chart(FIG_TREND_TRUE_VS_PRED)

    # figure trend barplot alerts
    col2_trend.plotly_chart(FIG_TREND_BARPLOT_ALERTS)

    # figure plotly traffic light alert
    #col3_trend.plotly_chart(FIG_TRAFFIC_LIGHT)
