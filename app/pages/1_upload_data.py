import streamlit as st
from datetime import datetime
from src import hello_world
import pandas as pd
import json
import gcsfs


# ---------------------------- read env variables used in the app ----------------------------
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_GCP = os.environ.get("PROJECT_GCP", "")
REGION_GCP = os.environ.get("REGION_GCP", "")
BUCKET_GCP = os.environ.get("BUCKET_GCP", "")


# ---------------------------- Connect to GCS  - gcsfs package ----------------------------
fs = gcsfs.GCSFileSystem(project = PROJECT_GCP)


# ---------------------------- Main Tittle ----------------------------
# Give your app a title
st.title("Upload Dataset to do Auto Data Science")


# ---------------------------- Upload file ----------------------------
st.divider()
st.header("Upload dataset")
st.write("Upload timeseries dataset with features and target to do a forecast 'x' intervals to future")

# use st.file_uploader to load files
uploaded_file = st.file_uploader("Upload a data file", type=["xlsx"])

# process the file when it is uploaded
if uploaded_file is not None:

    # read a excel file - only this file supported
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        data = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        st.write('file uploaded is not supported (not in excel format - xlsx)')


# show data if data exists
if 'data' in locals():
    st.write(data)


# ---------------------------- Define parameters (form) ----------------------------
st.divider()
st.header("Define Parameters to do Auto Data Science")
st.write("Define parameters such as: list of features, target, type of machine learning models to train, steps to future to do forecast, kind of exploratory data \
         analysis to do, etc")

# read list of features and target uploaded in the dataset
if 'data' in locals():
    list_features_target = data.columns.tolist()
    #st.write(list_features_target)
else:
    list_features_target = ["A", "B", "C", "D", "E"]


# use a form to define all the parameters that need the dataset
with st.form(key ='form_parameters'):
    
    # define base parameters . parameters always needed
    st.write('----- Base Parameters -----')
    NAME_DATASET = st.text_input("Define dataset name", "example_dataset")
    STEPS_FORECAST = st.number_input("Define number of steps to forecast", 5)
    LIST_FEATURES = st.multiselect("Select features", options = list_features_target[1:] , max_selections = len(list_features_target)-1)
    LIST_TARGET = st.multiselect("Select target", options = list_features_target[1:], max_selections = 1)

    # define parameters to EDA
    st.divider()
    st.write('----- EDA parameters -----')
    STATISTICS = st.checkbox("Statistics", value = False)
    HISTOGRAMS = st.checkbox("Histograms", value = True)
    BOXPLOTS_MONTLY = st.checkbox("Boxplots montly", value = True)
    CORRELATIONS_ALL = st.checkbox("Correlations all features", value = True)
    CORRELATIONS_TARGET = st.checkbox("Correlations target", value = True)
    SEGMENTATION_ANALYSIS = st.checkbox("Segmentation Analysis", value = True)
    #SEG_PARAM_TO_SEGMENT = st.multiselect("Select feature to segment data", options = list_features_target, max_selections = 1)
    SEG_PARAM_TO_SEGMENT = st.selectbox("Select feature to segment data", options = list_features_target[1:], index=1)
    CATEGORICAL_ANALYSIS = st.checkbox("Categorical Analysis", value = True)

    # define parameters to train the model
    st.divider()
    st.write('----- Machine Learning Parameters (by default all models trained) -----')
    

    # submit button
    st.divider()
    submitted1 = st.form_submit_button(label = 'Upload dataset')




# ---------------------------- Upload dataset file and parameter file ----------------------------
# After uploaded the file and complete the form with the parameters - upload them

if submitted1:
    # Create segmentations parameters according the user input
    threshold_1 = data[SEG_PARAM_TO_SEGMENT].min() - 10
    threshold_2 = data[SEG_PARAM_TO_SEGMENT].quantile(0.25)
    threshold_3 = data[SEG_PARAM_TO_SEGMENT].quantile(0.75)
    threshold_4 = data[SEG_PARAM_TO_SEGMENT].max() + 10
    SEG_DATA_INTERVALS = [threshold_1, threshold_2, threshold_3, threshold_4]
    SEG_DATA_LABELS = [SEG_PARAM_TO_SEGMENT + ' low', SEG_PARAM_TO_SEGMENT + ' medium', SEG_PARAM_TO_SEGMENT + ' high']

    # Create json/dictionary that will be saved with the parameters
    dict_parameters_data = {
        "steps_forecast": STEPS_FORECAST,
        "list_features": LIST_FEATURES,
        "list_target": LIST_TARGET,
        "eda":{
            "statistics":STATISTICS,
            "histograms":HISTOGRAMS,
            "boxplots_montly": BOXPLOTS_MONTLY,
            "correlations_all": CORRELATIONS_ALL,
            "correlations_target": CORRELATIONS_TARGET,
            "segmentation_analysis":SEGMENTATION_ANALYSIS,
            "seg_param_to_segment": SEG_PARAM_TO_SEGMENT,
            "seg_data_intervals": SEG_DATA_INTERVALS,
            "seg_data_labels": SEG_DATA_LABELS,
            "categorical_analysis": CATEGORICAL_ANALYSIS
            }
        }

    # Build path to save artifacts of this dataset
    path_gcs_folder_data = "gs://" + BUCKET_GCP + '/' + NAME_DATASET + '/' + 'data' + '/'

    # upload parameters locally
    path_local_parameters = 'data_local/parameters.json'
    with open(path_local_parameters, 'w') as file:
        json.dump(dict_parameters_data, file)

    # upload parameters cloud
    path_cloud_parameters = path_gcs_folder_data + 'parameters.json'
    with fs.open(path_cloud_parameters, 'w') as file:
        json.dump(dict_parameters_data, file)
    st.write('parameters uploaded')

    # upload data locally
    path_local_data = 'data_local/data.xlsx'
    data.to_excel(path_local_data)

    # upload data cloud
    path_cloud_data = path_gcs_folder_data + 'data.xlsx'
    data.to_excel(path_cloud_data)
    st.write('data uploaded')