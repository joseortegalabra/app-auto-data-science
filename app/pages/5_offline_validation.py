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
st.title("Offline Evaluation of Machine Learning Models")



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




# ---------------------------- Given a dataset selected - Given a model selected - see validation offline ----------------------------
st.divider()
st.header("See validation offline")

if 'NAME_DATASET' and 'SELECTED_RUN' in locals(): # if dataset and run are selected, show the information


    ######################## I) LOAD DATA AND MODELS ########################
    ### 1. Load model and dataset selected by the user


    ### 2. Load data train and test
    X_train, y_train, X_test, y_test = off_eval.load_data(BUCKET_GCP, NAME_DATASET, SELECTED_RUN)


    ### 3. Load model trained
    path_model = f'gs://{BUCKET_GCP}/{NAME_DATASET}/{SELECTED_RUN}/model.pkl'
    model = pd.read_pickle(path_model)


    ### 4. Load list features
    list_features = X_train.columns.tolist()
    list_target = y_train.columns.tolist()
    list_features_target = list_features + list_target



    ######################## III) OFFLINE EVALUATION MODEL ########################

    ##### format to show infomation
    # st.divider()
    # st.write('title of the plot')
    # st.pyplot(image to show)


    ### 0. Get predictions
    y_test_pred = off_eval.get_predictions(model, X_test, y_test)


    ### 1. Metrics
    #### 1.1 Metric for Model trained
    st.divider()
    st.write('Metrics of the machine learning model trained')
    metrics = off_eval.calculate_metrics_regressors_models(y = y_test, 
                                                y_pred = y_test_pred, 
                                                model_name = SELECTED_RUN, 
                                                decimals_round = None
                                                )
    st.write(metrics)


    #### 1.2 Basic Model - mean target / predict common class
    st.divider()
    st.write('Metrics of baseline model - predict the mean value')

    # calculate mean y train. ADJUST BASIC MODEL
    y_basic_model = y_train.mean().values[0]

    # generate vector to y_pred to evaluate. obs generate len according the y_true when the mean prediction will be compared
    y_basic_model_pred = pd.DataFrame(y_basic_model * np.ones([y_test.shape[0]]))
    y_basic_model_pred.index = y_test.index
    y_basic_model_pred.columns = y_test.columns

    # metrics basic model
    metrics_basic_model = off_eval.calculate_metrics_regressors_models(y = y_test, 
                                                            y_pred = y_basic_model_pred, 
                                                            model_name = 'd0eop_microkappa//Basic Model Pred', 
                                                            decimals_round = None
                                                            )

    st.write(metrics_basic_model)


    ### 2. Plot Predictions
    #### 2.1 y_true vs y_pred
    st.divider()
    st.write('Plot y_true vs y_pred')
    y_true_y_pred = off_eval.plot_y_true_vs_y_pred(y = y_test, 
                                        y_pred = y_test_pred, 
                                        title_plot = SELECTED_RUN
                                        )
    st.pyplot(y_true_y_pred)


    #### 2.2 hist errors
    st.divider()
    st.write('Plot histogram of errors')
    hist_errors = off_eval.hist_errors_predictions(y = y_test, 
                                        y_pred = y_test_pred, 
                                            title_plot = SELECTED_RUN
                                    )
    st.pyplot(hist_errors)


    ### 3. Explanaible AI
    #### 3.1 Permutation Importances
    st.divider()
    st.write('Plot Permutation Importances')
    plot_permutation_importances, df_importances = off_eval.permutation_importances(model, list_features, X_test, y_test)
    st.pyplot(plot_permutation_importances)


    #### 3.2 Permutation Importances with Noise Features
    st.divider()
    st.write('Plot Permutation Importances with noise')
    plot_permutation_importances_noise = off_eval.permutation_importances_noise(model, X_train, y_train, X_test, y_test)
    st.pyplot(plot_permutation_importances_noise)


    #### 3.3 Partial Dependence Plot
    st.divider()
    st.write('Partial Depence Plot')
    fig_pdp = off_eval.pdp_plot(model, X_test)
    st.pyplot(fig_pdp)


    ### 4. Perturbation test - Sensitivy Analysis
    #### 4.1 Peturbation test one feature global plots
    # get the most important feature
    st.divider()
    st.write('Perturbation test - Sensitivy Analysis')

    MOST_IMPORTANT_FEATURE = df_importances.sort_values(ascending = False).index.values[0]
    fig_sensitivy_one_feature = off_eval.perturbation_test_one_feature_global_analysis(tag_sensitivy_analysis = MOST_IMPORTANT_FEATURE, # most important feature for example
                                                                                    epsilon = 1, 
                                                                                    model = model,
                                                                                    X = X_test.copy(), 
                                                                                    y = y_test.copy(), 
                                                                                    y_pred = y_test_pred.copy()
                                                                                    )

    st.pyplot(fig_sensitivy_one_feature)


    ### 5. Metrics by segments
    #### 5.1 Segment by most important feature
    st.divider()
    st.write('Metrics ML model - divided by segment of the most important feature')
    tag_segment = MOST_IMPORTANT_FEATURE
    metrics_segmentation = off_eval.metrics_segmentation_analysis(tag_segment, X_test, y_test, y_test_pred, metrics)
    st.write(f'Most Important Feature: {MOST_IMPORTANT_FEATURE}')
    st.write(metrics_segmentation)