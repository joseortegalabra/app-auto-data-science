import streamlit as st
from datetime import datetime
from src import hello_world
from src import get_dataset
from src import eda

import pandas as pd
import gcsfs
import json


# ---------------------------- read env variables used in the app ----------------------------
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_GCP = os.environ.get("PROJECT_GCP", "")
REGION_GCP = os.environ.get("REGION_GCP", "")
BUCKET_GCP = os.environ.get("BUCKET_GCP", "")


# ---------------------------- Auxiliar functions ----------------------------
def read_data_file(BUCKET_GCP, NAME_DATASET):
    """
    read datafile
    """
    # read data
    path_gcs_df = f'gs://{BUCKET_GCP}/{NAME_DATASET}/data/data.xlsx'
    df = pd.read_excel(path_gcs_df)

    # correction data read gcs
    try:
        df = df.drop(columns = 'Unnamed: 0')
    except:
        pass
    
    # set index
    df = df.set_index('Date', drop = True)

    return df

def read_json_config(bucket_gcp, name_dataset):
    """
    Read json config
    """
    # connect to GCS as pythonic way
    fs = gcsfs.GCSFileSystem()
    
    # path json
    path_gcs_json = f'gs://{bucket_gcp}/{name_dataset}/data/parameters.json'
    
    # read json
    with fs.open(path_gcs_json, 'r') as file:
        dict_parameters_data = json.load(file)
    
    return dict_parameters_data



# ---------------------------- Main Tittle ----------------------------
# Give your app a title
st.title("Exploratory Data Analysis")


# ---------------------------- Search list of datasets and select one to do EDA ----------------------------
st.divider()
st.header("Select dataset")
#st.write("Select one of the dataset saved")

# load list of dataset loaded
list_id_datasets = get_dataset.read_name_folders_bucket_gcs(bucket_name = BUCKET_GCP)
with st.form(key ='form_datasets'):
    NAME_DATASET = st.selectbox("Select dataset loaded", options = list_id_datasets, index=0)
    submitted1 = st.form_submit_button(label = 'Select dataset')




# ---------------------------- EDA (when a dataset is selected in the form) ----------------------------
st.divider()
st.header("EDA")

if submitted1:
    ### 2. Load files of the case according the name of the dataset - id
    #### 2.1 Read data
    st.write('-- Data --')
    data = read_data_file(BUCKET_GCP, NAME_DATASET)
    st.write(data.head())


    #### 2.2 Read json configuration
    dict_parameters_data = read_json_config(BUCKET_GCP, NAME_DATASET)

    # list features
    list_target = dict_parameters_data['list_target']
    list_features = dict_parameters_data['list_features']

    # parameters to indicate which plot will be show
    show_statistics = dict_parameters_data['eda']['statistics']
    show_histograms = dict_parameters_data['eda']['histograms']
    show_boxplots_montly = dict_parameters_data['eda']['boxplots_montly']
    show_correlations_all = dict_parameters_data['eda']['correlations_all']
    show_correlations_target = dict_parameters_data['eda']['correlations_target']
    show_segmentation_analysis = dict_parameters_data['eda']['segmentation_analysis']
    show_categorical_analysis = dict_parameters_data['eda']['categorical_analysis']

    # parameters segmentation
    seg_param_to_segment = dict_parameters_data['eda']['seg_param_to_segment']
    seg_data_intervals = dict_parameters_data['eda']['seg_data_intervals']
    seg_data_labels = dict_parameters_data['eda']['seg_data_labels']



    ### 3. Functions of EDA
    #### 3.1 statistics
    if show_statistics:
        st.write('-- Statistics --')
        df_statistics = eda.generate_descriptive_statistics(data)
        st.write(df_statistics)


    #### 3.2 Histograms
    if show_histograms:
        st.write('-- Histograms --')
        fig_sns_kde_hist = eda.plot_kde_hist(data)
        fig_sns_kde_hist


    #### 3.3 Boxplots monthly
    if show_boxplots_montly:
        st.write('-- Boxplots montly --')
        fig_boxplot_months = eda.plot_multiple_boxplot_months(data)
        st.plotly_chart(fig_boxplot_months)



    #### 3.4 Correlations - all
    if show_correlations_all:
        st.write('-- Correlations all --')
        _, df_corr_upper = eda.calculate_correlations_triu(data)
        fig_corr = eda.plot_heatmap(df_corr_upper)
        fig_corr



    #### 3.5 Correlations - target
    if show_correlations_target:
        st.write('-- Correlations target --')
        corr_target = eda.calculate_correlations_target(df = data, target = list_target[0])
        fig_corr_target = eda.plot_heatmap(corr_target)
        st.plotly_chart(fig_corr_target)



    #### 3.6 Segmentation - Selected feature to segment the data
    if show_segmentation_analysis:
        # define variable name used to segmentation
        var_segment_name = seg_param_to_segment + '_segments'
        
        # generate data segmentation
        data_segmented = eda.custom_segmentation(df = data.copy(),
                                                var_segment = seg_param_to_segment, 
                                                intervals_segments = seg_data_intervals, 
                                                labels_segments = seg_data_labels
                                                )
        


    #### 3.7 Segmentation - freq
    if show_segmentation_analysis:
        st.write('-- Segmentation Analysis - Frequency --')
        fig_freq_segmentation = eda.plot_freq_segmentation(df = data_segmented, var_segment = var_segment_name)
        fig_freq_segmentation



    #### 3.8 Segmentation - boxplot
    if show_segmentation_analysis:
        st.write('-- Segmentation Analysis - Boxplots --')
        fig_boxplot_segments = eda.plot_boxplots_segments(df = data_segmented, var_segment = var_segment_name)
        fig_boxplot_segments



    #### 3.9 Segmentation - corr - all
    if show_segmentation_analysis:
        st.write('-- Segmentation Analysis - Correlations all --')
        corr_segments = eda.calculate_correlations_triu_segmentation(df = data_segmented, var_segment = var_segment_name)
        fig_corr_segments = eda.plot_corr_segmentation_subplots_heatmap(corr_segments)
        fig_corr_segments



    #### 3.10 Segmentation - corr - target
    if show_segmentation_analysis:
        st.write('-- Segmentation Analysis - Correlations target --')
        corr_segments_target = eda.calculate_correlations_target_segmentation(df = data_segmented, 
                                                                            var_segment = var_segment_name, 
                                                                            target = list_target[0]
                                                                            )
        fig_corr_segments_target = eda.plot_corr_segmentation_subplots_heatmap(corr_segments_target)
        fig_corr_segments_target



    #### 3.11 Categorical Analysis - genarate categorical feautures
    # generate data percentile
    if show_categorical_analysis:
        data_percentile_feature_target = data.copy()
        for index, variable in enumerate(list_features + list_target):
            data_percentile_feature_target = eda.percentile_segmentation(df = data_percentile_feature_target, 
                                                                        var_segment = variable, 
                                                                        type_percentile = "quartile"
                                                                        )
            data_percentile_feature_target.drop(columns = variable, inplace = True)



    #### 3.12 Categorical Analysis - freq each feature againts freq target
    if show_categorical_analysis:
        st.write('-- Categorical Analysis - Frequency --')
        df_statics_categorical_data = eda.descriptive_statistics_target_for_each_feature(df = data_percentile_feature_target, 
                                                                                        target = 'quartile_' + list_target[0])
        df_statics_categorical_data


    #### 3.13 Categorical Analysis - crosstab freq target vs each feature
    if show_categorical_analysis:
        st.write('-- Categorical Analysis - Crosstab --')
        fig_crosstab_categorical = eda.barplot_crosstab_freq_target_1_features(df = data_percentile_feature_target,
                                                                            target = 'quartile_' + list_target[0])
        fig_crosstab_categorical
