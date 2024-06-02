"""
EXTRA CODES - USED ONLY TO TEST WAYS TO SELECT DATASET AND MODELS
"""

import streamlit as st
from datetime import datetime
from src import hello_world
from src import get_dataset
from src import info_vertex_experiment as info_exp
from google.cloud import aiplatform as vertex_ai

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# offline evaluation codes
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from src import offline_evaluation as off_eval



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
st.title("Test choices to select dataset and models")




#####################################################################################################################################################################################
#####################################################################################################################################################################################
# ---------------------------- WAY 1 - SEACH DATASET AND THE MODEL - EACH TIME THAT THE USER CLICK ----------------------------
st.header('WAY 1 - SEACH DATASET AND THE MODEL - EACH TIME THAT THE USER CLICK')

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

# ------print output ------
st.subheader('SELECTIONS')
st.write(f'DATASET SELECTED: {NAME_DATASET}')
st.write(f'MODEL SELECTED: {SELECTED_RUN}')






######################################################################################################################################################################################
######################################################################################################################################################################################
# ---------------------------- WAY 2 - SEACH DATASET AND THE MODEL - USE A FORM TO SELECT DATASET AND THEN SELECT THE MODEL ----------------------------
st.header('WAY 2 - SEACH DATASET AND THE MODEL - USE A FORM TO SELECT DATASET AND THEN SELECT THE MODEL')

with st.form(key ='dataset'):
    # ------ load list of dataset loaded ------
    list_id_datasets = get_dataset.read_name_folders_bucket_gcs(bucket_name = BUCKET_GCP)
    NAME_DATASET = st.selectbox("Select dataset loaded", options = list_id_datasets, index=0)  # <-------------------
    
    # select dataset
    submitted_dataset = st.form_submit_button(label = 'Select dataset')

with st.form(key ='model'):
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
    
    # select model
    submitted_model = st.form_submit_button(label = 'Select model')


# ------print output ------
st.subheader('SELECTIONS')
st.write(f'DATASET SELECTED: {NAME_DATASET}')
st.write(f'MODEL SELECTED: {SELECTED_RUN}')

