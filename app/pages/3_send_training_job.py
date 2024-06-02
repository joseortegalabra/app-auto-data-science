import streamlit as st
from src import hello_world
from src import get_dataset

import datetime as dt
import pandas as pd
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip


# ---------------------------- read env variables used in the app ----------------------------
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_GCP = os.environ.get("PROJECT_GCP", "")
REGION_GCP = os.environ.get("REGION_GCP", "")
BUCKET_GCP = os.environ.get("BUCKET_GCP", "")
MAI_SA = os.environ.get("MAIL_SA", "")


# ---------------------------- fff ----------------------------



# ---------------------------- Main Tittle ----------------------------
# Give your app a title
st.title("Send training job in vertex to train several models")


# ---------------------------- Search list of datasets and select one to do EDA ----------------------------
st.divider()
st.header("Select dataset")
#st.write("Select one of the dataset saved")

# load list of dataset loaded
list_id_datasets = get_dataset.read_name_folders_bucket_gcs(bucket_name = BUCKET_GCP)
with st.form(key ='form_datasets'):
    NAME_DATASET = st.selectbox("Select dataset loaded", options = list_id_datasets, index=0)
    submitted1 = st.form_submit_button(label = 'Select dataset')


# ---------------------------- Given a dataset selected - send training job ----------------------------
st.divider()
st.header("Send trainning job")

if submitted1:

    ### ------------- Paso 0. Parámetros generales -------------

    # definir un bucket (ya creado) para guardar los archivos que genera el usar VERTEX AI.
    BUCKET_ID = f'{BUCKET_GCP}/vertex-ai-jobs'

    # obtener la hora actual de cuándo se comenzó la ejecución - hash
    now = dt.datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    # identificacion del tipo de caso de uso (y también tipo de modelo) que se va a usar poara registrar el entrenamiento
    identity_kind_use_case = 'auto-data-science-job'  

    # definir path donde se va a guardar el pkl con el modelo y que además quede registrado en modelos de vertex
    path_artifact_model_vertex = f'gs://{BUCKET_GCP}/vertex-ai-registry-model/{NAME_DATASET}/run_{date_time}/'


    st.write('Parámetros Generales GCP')
    st.write('PROJECT_GCP: ', PROJECT_GCP)
    st.write('BUCKET_ID: ', BUCKET_ID)
    st.write('REGION_GCP: ', REGION_GCP)

    st.write('\n----')
    st.write('NAME DATASET - USE CASE - EXPERIMENT: ', NAME_DATASET)

    st.write('\n----')
    st.write('Parámetros Específicos job entrenamiento')
    st.write('date_time: ', date_time)
    st.write('identity_kind_use_case: ', identity_kind_use_case)
    st.write('path_artifact_model_vertex: ', path_artifact_model_vertex)



    ### ------------- Paso 1. Crear script de entrenamiento -------------
    #done

    ### ------------- Paso 2: Inicializar Vertex AI -------------
    aiplatform.init(project = PROJECT_GCP, location = REGION_GCP, staging_bucket = BUCKET_ID)



    ### ------------- Paso 3. Definir parámetros necesarios para CREAR la instancia del job de entrenamiento -------------
    ### definir el nombre del job que se enviará. Algo que indentifique de qué es el job + hora envio ###
    job_name = identity_kind_use_case + '__job_train__' + date_time
    st.write('job name: ', job_name)

    ### definir containers
    container_train = 'us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest'
    container_deploy = 'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest'

    ### definir el path al script de entrenamiento ###
    path_train_script = 'src/task_train_models.py'   # path according the root of the app

    ### definir la descripción del modelo ###
    description = 'entrenar modelo leyendo pkl de GCS'

    ### definir los requirements ###
    list_requirements = ["pandas", "matplotlib", "seaborn", "plotly", "numpy", "scikit-learn", "python-dotenv", "gcsfs", \
                        "joblib", "openpyxl", "google-cloud-bigquery", "db-dtypes", "google-cloud-aiplatform"]


    ### ------------- Paso 4. Definir parámetros necesarios para ENVIAR job de entrenamiento - usando CPU -------------
    ### definir el nombre con el que queda registrado (en VERTEX AI) el modelo resultado del entrenamiento ###
    model_name = identity_kind_use_case  + '__model__' + date_time 
    st.write('model name: ', model_name)

    ### definir el tipo de máquina para hacer el entrenamiento ###
    machine_type_train = "n1-standard"
    vcpu_train = "4"
    train_compute = machine_type_train + "-" + vcpu_train
    st.write("Train machine type: ", train_compute)

    
    ### ------------- Paso 5. Crear instancia del job de entrenamiento a VERTEX AI (CustomTrainingJob) -------------
    job = aiplatform.CustomTrainingJob(
        display_name = job_name,
        script_path = path_train_script,
        model_description = description,
        container_uri = container_train,
        requirements = list_requirements,
        model_serving_container_image_uri = container_deploy,
    )
    st.write('class job: ', job)



    ### ------------- Paso 6. Enviar el job de entrenamiento a VERTEX AI (CustomTrainingJob) -------------
    # - Importante 4. Para que **no aparesca el texto que está corriendo el job y se pueda seguir utilizando el código se puede utilizar el parámetro "sync" y setear en "False"**

    # add SA
    model = job.run(
        model_display_name = model_name,
        replica_count = 1,
        machine_type = train_compute,
        base_output_dir = path_artifact_model_vertex, # path custom .../model/model.pkl donde se guarda el pkl del modelo. se omite del path model/model.pkl
        args = ["--id_date_time=" + date_time, "--name_dataset=" + NAME_DATASET], # args que se le pasan al script de entrenamiento de este ejemplo
        sync = False, # doesnt show training
        service_account = f"{MAI_SA}@{PROJECT_GCP}.iam.gserviceaccount.com"
    )