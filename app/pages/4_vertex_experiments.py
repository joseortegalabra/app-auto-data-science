from src import hello_world
from src import get_dataset
from src import info_vertex_experiment as info_exp
from google.cloud import aiplatform as vertex_ai

import streamlit as st
from datetime import datetime
import pandas as pd
import gcsfs
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# ---------------------------- read env variables used in the app ----------------------------
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_GCP = os.environ.get("PROJECT_GCP", "")
REGION_GCP = os.environ.get("REGION_GCP", "")
BUCKET_GCP = os.environ.get("BUCKET_GCP", "")


# ---------------------------- fff ----------------------------




# ---------------------------- Main Tittle ----------------------------
# Give your app a title
st.title("Metrics of Tranning of Machine Learning Models - Vertex Experiments")



# ---------------------------- Search list of datasets / search list of models ----------------------------
st.divider()
st.header("Select dataset")
#st.write("Select one of the dataset saved")

# ------ load list of dataset loaded ------
list_id_datasets = get_dataset.read_name_folders_bucket_gcs(bucket_name = BUCKET_GCP)
NAME_DATASET = st.selectbox("Select dataset loaded", options = list_id_datasets, index=0)  # <-------------------


# ------ load list of models trainned ------
# params
EXPERIMENT_NAME = NAME_DATASET
EXPERIMENT_DESCRIPTION = f'Run forecasting models of a target. Dataset: {EXPERIMENT_NAME}'

# search tensorboard instance, if it doesn't exist -> created it
id_tensorboard_vertex = info_exp.get_tensorboard_instance_or_create(experiment_name = EXPERIMENT_NAME,
                                                                    experiment_description = EXPERIMENT_DESCRIPTION,
                                                                    project_gcp = PROJECT_GCP,
                                                                    location_gcp = REGION_GCP
                                                                   )

# set experiment (or created if it doesn't exist - automatically)
print('\n--- setting experiment vertex ai ---')
vertex_ai.init(
    experiment = EXPERIMENT_NAME,
    experiment_description = EXPERIMENT_DESCRIPTION,
    experiment_tensorboard = id_tensorboard_vertex,
    project = PROJECT_GCP,
    location = REGION_GCP,
    )

# get a list of all runs in the experiment
df_results_experiments = vertex_ai.get_experiment_df(EXPERIMENT_NAME) # it takes some seconds
list_runs = df_results_experiments['run_name'].tolist()
SELECTED_RUN = st.selectbox("Select model trained", options = list_runs, index=0) # <-------------------



# ---------------------------- Given a dataset selected - Given a model selected - see info training vertex experiments ----------------------------
st.divider()
st.header("See info of training saved vertex experiment")

if 'NAME_DATASET' and 'SELECTED_RUN' in locals(): # if dataset and run are selected, show the information

    ##### get information of the selected RUN
    params_trainning, params_problem, metrics_trainning, img_true_pred = info_exp.show_information_run(SELECTED_RUN, EXPERIMENT_NAME)

    ##### format to show infomation
    # st.divider()
    # st.write('title of the plot')
    # st.pyplot(image to show)

    # context
    st.divider()
    st.write('show params context problem')
    params_problem


    # models
    st.divider()
    st.write('show params models')
    params_trainning

    # metrics
    st.divider()
    st.write('show metrics')
    metrics_trainning


    # ytrue vs ypred
    st.divider()
    st.write('show plot ytrue vs ypred')
    plt.imshow(img_true_pred)
    plt.axis('off')
    fig_ytrue_ypred = plt.gcf() # get actual plt figure
    plt.close() # close plt
    st.pyplot(fig_ytrue_ypred)