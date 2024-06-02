import streamlit as st
from datetime import datetime
from src import hello_world
from src import get_dataset

from src import info_vertex_experiment as info_exp
from google.cloud import aiplatform as vertex_ai

import pandas as pd
import numpy as np
import gcsfs
import json

# ---------------------------- set wide mode ----------------------------
#st.set_page_config(layout = 'wide')


# ---------------------------- read env variables used in the app ----------------------------
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_GCP = os.environ.get("PROJECT_GCP", "")
REGION_GCP = os.environ.get("REGION_GCP", "")
BUCKET_GCP = os.environ.get("BUCKET_GCP", "")


# ---------------------------- Main Tittle ----------------------------
# Give your app a title
st.title("Simulations with Machine Learning Models")


# PARAMETERS BY THE USER - IN THIS EXAMPLE ARE FIXED
NAME_DATASET = 'develop-app-final-v2'
SELECTED_RUN = 'run-lr'



# ---------------------------- TRANSVERSAL CODE ----------------------------

#########################  I) INITIALIZE #########################
### 1. Define dataset and model
#

### 2. Load data train
# example path
# f'gs://{bucket_gcs}/{experiment_name}/{run_name}/{path_artifact_locally}'

# X_train
path_X_train = f'gs://{BUCKET_GCP}/{NAME_DATASET}/{SELECTED_RUN}/X_train.pkl'
X_train = pd.read_pickle(path_X_train)

# y_train
path_y_train = f'gs://{BUCKET_GCP}/{NAME_DATASET}/{SELECTED_RUN}/y_train.pkl'
y_train = pd.read_pickle(path_y_train)

### 3. Load model trained
path_model = f'gs://{BUCKET_GCP}/{NAME_DATASET}/{SELECTED_RUN}/model.pkl'
model = pd.read_pickle(path_model)

### 4. Load list features
list_features = X_train.columns.tolist()
list_target = y_train.columns.tolist()
list_features_target = list_features + list_target


#########################  II) SIMULATIONS #########################
#### 1.2 Calculate min and max date to get the initial value
# min date
min_date = X_train.index.min()

# max date
max_date = X_train.index.max()

# median date
index_median = round(X_train.index.shape[0] / 2, 0)
median_date = X_train.iloc[[index_median], :].index[0]


# ---------------------------- SIMULATION ONE - SELECT DATES IN THE CALENDER ----------------------------
st.divider()
st.header("Simulation 1: select dates in the calender")

#### 1.3 Given a certain date, get the initial values
#INITIAL_DATE = '2021-02-27'
INITIAL_DATE = st.date_input('date simulation', value = median_date, min_value = min_date, max_value = max_date)


#### 1.4 Get Initial Instance to do Inference
# try get the row of the initial date - if the date doesn't exist, search the most close date
initial_date_to_inference = pd.Timestamp(INITIAL_DATE)
try:
    initial_instance = X_train.loc[[initial_date_to_inference]]
except KeyError:
    closest_date = X_train.index[np.abs((X_train.index - initial_date_to_inference)).argmin()]
    initial_instance = X_train.loc[[closest_date]]


st.write('#### Instance')
st.write(f'Date: {initial_date_to_inference}')
st.write(f'Features: ')
st.dataframe(initial_instance)

### 2. Predict with the initial value
#### 2.1 Predict with the initial value
prediction = model.predict(initial_instance)

#### 2.2 Show the true value of this instance
# try get the row of the initial date - if the date doesn't exist, search the most close date
try:
    y_true = y_train.loc[[initial_date_to_inference]].values
except KeyError:
    closest_date = y_train.index[np.abs((y_train.index - initial_date_to_inference)).argmin()]
    y_true = y_train.loc[[closest_date]].values

st.write('#### Predictions')
st.write(f'Prediction: {prediction}')
st.write(f'True value: {y_true}')



# ---------------------------- SIMULATION TWO: CHANGE VALUES OF THE FEATURES ----------------------------
st.divider()
st.header('Simulation 2: change values of the features')


#### 3.1 Save previos prediction in session state. The logic is: save previous values with the actual values, predict, get actual values. 
# This work using "memory" - session states
st.session_state['prev_df_simulation'] = st.session_state['df_simulation']
st.session_state['prev_prediction_simulation'] = st.session_state['prediction_simulation']

#### 3.2 Calculate min and max values for each feature
min_values_features = X_train.min()
max_values_features = X_train.max()

#### 3.X Generate 3 columns: input values, last prediction, actual prediction
simulation_col1, simulation_col2, simulation_col3, simulation_col4 = st.columns(4)

#### 3.3 Create initial input values
simulation_col1.write('#### Input features')
dict_input_values_features = {}
for var_name in list_features:
    dict_input_values_features[var_name] = simulation_col1.slider(var_name,
                                                    min_value = min_values_features[var_name], 
                                                    max_value = max_values_features[var_name], 
                                                    #value = X_train.loc[median_date][var_name] # initial values of the mediana of the data no sorted
                                                    value = initial_instance[var_name].values[0] # initial value is the value selected by the user at the start of the codes
                                                    )

##### 3.4 Generate input dataframe with the input features
# generate dataframe with the list of input values
list_values_simulation = []
for var_name in list_features:
    list_values_simulation.append(dict_input_values_features[var_name])
df_simulation = pd.DataFrame([list_values_simulation], columns = list_features)

#### 3.5 Predict - NOTE THAT IN THIS CASE THERE ARE NO TRUE VALUE TO COMPARE
prediction_simulation = model.predict(df_simulation)

#### 3.6 Save actual prediction in session state
st.session_state['df_simulation'] = df_simulation.T
st.session_state['prediction_simulation'] = prediction_simulation

### 3.8 Calculate difference
diff_df_simulation = st.session_state['df_simulation'] - st.session_state['prev_df_simulation']
diff_prediction_simulation = st.session_state['prediction_simulation'] - st.session_state['prev_prediction_simulation']

#### 3.7 Show outputs predictions - features and prediction
simulation_col2.write('#### -- last prediction --')
simulation_col2.dataframe(st.session_state['prev_df_simulation'])
simulation_col2.write(st.session_state['prev_prediction_simulation'])

simulation_col3.write('#### -- actual prediction --')
simulation_col3.dataframe(st.session_state['df_simulation'])
simulation_col3.write(st.session_state['prediction_simulation'])

simulation_col4.write('#### -- difference actual vs prev --')
simulation_col4.dataframe(diff_df_simulation)
simulation_col4.dataframe(diff_prediction_simulation)