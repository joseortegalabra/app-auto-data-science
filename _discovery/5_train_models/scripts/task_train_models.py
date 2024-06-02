# PARAMETERS THAT SHOULD BE READ FROM ENV VARIABLES OR PASSED AS ARGS
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_GCP = os.environ.get("PROJECT_GCP", "")
REGION_GCP = os.environ.get("REGION_GCP", "")
BUCKET_GCP = os.environ.get("BUCKET_GCP", "")
MAI_SA = os.environ.get("MAIL_SA", "")



import argparse

import pandas as pd
import numpy as np
import gcsfs
import json
import matplotlib.pyplot as plt
import joblib
import pickle
import os

# processing data
from sklearn.model_selection import train_test_split

# models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# evaluate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# vertex gcp
from google.cloud import aiplatform as vertex_ai
from google.cloud import storage


" ------------------- 0. READ ARGS PASSED TO THE SCRIPT ------------------- "

# Create parser and define the args
parser = argparse.ArgumentParser(description='Script to train ML models')
parser.add_argument('--name_dataset', type=str, help = 'Name of the dataset. The ID used to map the actual training')
parser.add_argument('--id_date_time', type=str, help='datetime ID pasado al argumento. datetime pasado de la hora que comienza a enviarse el job')

# Parse the arguments in the console
args = parser.parse_args()

# Save args as python variables in this script. "GLOBAL VARIABLES"
NAME_DATASET = args.name_dataset
date_time = args.id_date_time



" ------------------- 1. AUXILIAR FUNCTIONS ------------------- "
#### AUXILIAR FUNCTIONS READ DATA AND PARAMS
def read_data_file(bucket_gcp, name_dataset):
    """
    read datafile
    """
    # read data
    path_gcs_df = f'gs://{bucket_gcp}/{name_dataset}/data/data.xlsx'
    df = pd.read_excel(path_gcs_df)
    
    # set index
    df = df.set_index('Date')

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



##### AUXILIAR FUNCTIONS - evaluate models
def evaluate_model(y_true, y_predicted):
    """
    Given "y_true" and "y_predicted" calculate metrics of performance (r2, rmse, mae)
    """
    r2_metric = r2_score(y_true, y_predicted)

    rmse_metric = mean_squared_error(y_true, y_predicted, squared = False)

    mae_metric = mean_absolute_error(y_true, y_predicted)

    print("r2: ", r2_metric)
    print("rmse: ", rmse_metric)
    print("mae_metric: ", mae_metric)
    return r2_metric, rmse_metric, mae_metric


def plot_y_true_vs_y_pred(y, y_pred, title_plot):
    """
    Plot y_true vs y_pred (using matplotlib figure). y_true in X-axis, y_pred in Y-axis.

    Args:
        y (dataframe): dataframe with y-true values 
        y_pred (dataframe): dataframe with y-pred values
        title_plot (string): tittle in the plot
    
    Return
        fig (figure matplolib): figure to show, download, etc
    """
    fig, ax = plt.subplots()
    scatter_plot = ax.scatter(y, y_pred, alpha=0.3, marker='x', label='y_true vs y_pred')

    # Add bisectriz
    y_bisectriz = x_bisectriz = np.linspace(y.min()[0], y.max()[0], y.shape[0])
    ax.plot(x_bisectriz, y_bisectriz, label='Bisectriz', color='red', alpha=0.3)

    # Add names to axis
    ax.set_xlabel('Y true')
    ax.set_ylabel('Y pred')
    
    ax.set_title(title_plot)
    ax.legend()


    # save fig, return the local path and close fig
    name_y_true_y_pred = 'y_true_y_pred.png'
    plt.savefig(name_y_true_y_pred)
    plt.close()
    
    return fig, name_y_true_y_pred



##### AUXILIAR FUNCTIONS - registry experiments in vertex
def create_instance_tensorboard(experiment_name, experiment_description, project_gcp, location_gcp):
    """
    Create a vertex tensorboard instance. The instance of tensorboard is created with the idea to have the same name of the experiment of vertex ai
    that will use this instance of vertex tensorboard.

    Obs: This code create always a tensorboard instance, with the same name (display_name) but different ID, so it is necessary RUN ONCE
    
    Args
        experiment_name (string)
        experiment_description (string)
        project_gcp (string)
        location_gcp (string)

    Return
        id_experiment_tensorboard (vertex ai tensorboard object)
    """
    id_tensorboard_vertex = vertex_ai.Tensorboard.create(display_name = f'tensorboard-{experiment_name}',
                                                          description = f'tensorboard-{experiment_description}',
                                                          project = project_gcp,
                                                          location = location_gcp
                                                         )
    return id_tensorboard_vertex

def get_tensorboard_instance_or_create(experiment_name, experiment_description, project_gcp, location_gcp):
    """
    Search if exist a tensorboard instance and get it. If the instance doesn't exist, create it.
    The instance of tensorboard has its name with the idea to have the same name of the experiment of vertex ai that will use this instance
    of vertex.

    Args
        experiment_name (string)
        experiment_description (string)
        project_gcp (string)
        location_gcp (string)

    Return
        id_experiment_tensorboard (vertex ai tensorboard object)
    """
    
    ''' search tensorboard instance. if the list is empty the tensorboard instance doesn't exist and it will created '''
    # GET tensorboard instance created FILTERING by display name. return a list of the instance doesn't exist return a empty list
    list_tensorboard_vertex = vertex_ai.Tensorboard.list(
        filter = f'display_name="tensorboard-{experiment_name}"',
        project = project_gcp,
        location = location_gcp
    )

    # if vertex tensorboard instance doesn't exist, create it
    if len(list_tensorboard_vertex) == 0:
        print('--- creating vertex tensorboard instance ---')
        id_tensorboard_vertex = vertex_ai.Tensorboard.create(display_name = f'tensorboard-{experiment_name}',
                                                                 description = f'tensorboard-{experiment_description}',
                                                                 project = project_gcp,
                                                                 location = location_gcp
                                                                ) # return tensorboard instance created
    else:
        print('--- tensorboard instance already exists ---')
        id_tensorboard_vertex = list_tensorboard_vertex[0] # tensorboard instance exists, return it
    
    return id_tensorboard_vertex

def save_local_to_gcs(uri_gcs, uri_local):
    """
    AUXILIAR. Save a locally file onto GCS.
    Args:
        uri_gcs (string): path in gcs where the local file will be saved
        uri_local (strring). path in local where the local file was saved

    Return
        nothing
    """

    blob = storage.blob.Blob.from_string(uri_gcs, client=storage.Client())
    blob.upload_from_filename(uri_local)

def save_artifacts_experiments_vertex(path_artifact_locally, type_artifact, bucket_gcs, experiment_name, run_name):
    """
    Save an artifact in experiments in vertex. This functions works for an individual artifact. The run of the experiment needs to be created
    The input is a file saved locally and the output is the file registered as a artifact of a run of a vertex experiment
    
    There following steps are necesarys to save the artifact
    - save artifact locally
    - save artifact in GCS
    - link the artifact in GCS with vertex metadata
    - link vertex metadata with an artifact saved in a run (experiment vertex)
    - delete the file locally
    """

    # 1. save artifact locally (done -input function)


    # 2. save artifact in GCS
    path_artifact_gcs = f'gs://{bucket_gcs}/{experiment_name}/{run_name}/{path_artifact_locally}'
    save_local_to_gcs(uri_gcs = path_artifact_gcs, 
                      uri_local = path_artifact_locally)

    
    # 3. link the artifact in GCS with vertex metadata
    path_artifact_locally_corrected = path_artifact_locally.replace('_', '-').replace('.', '-') # in the name only accepted "-"
    path_artifact_locally_corrected = path_artifact_locally_corrected.lower() # in the name only acceted lower case [a-z0-9][a-z0-9-]{0,127}
    
    
    artifact_metadata = vertex_ai.Artifact.create(
        schema_title = "system.Artifact", 
        uri = path_artifact_gcs, # 
        display_name = f"artifact-{path_artifact_locally}", # nombre con el que se muestra en el menu "metadata"
        description = f"description-{path_artifact_locally}",
        resource_id = f"{path_artifact_locally_corrected}-{experiment_name}-{run_name}"  # nombre con el que se muestra en el menu "artifact del run del experimento" de vertex. No acepta espacios
        )


    # 4. link vertex metadata with an artifact saved in a run 
    executions = vertex_ai.start_execution(
        schema_title="system.ContainerExecution", 
        display_name='REGISTRO DE ARTIFACTS'
    )
    executions.assign_input_artifacts([artifact_metadata])

    
    # 5. delete the file local
    #os.remove(path_artifact_locally)




" ------------------- 2. LOAD FILES ACCORDING THE NAME DATASET ------------------- "
#### 2.1 Read data
data = read_data_file(BUCKET_GCP, NAME_DATASET)


#### 2.2 Read json configuration
dict_parameters_data = read_json_config(BUCKET_GCP, NAME_DATASET)
list_target = dict_parameters_data['list_target']
list_features = dict_parameters_data['list_features']
steps_forecast = dict_parameters_data['steps_forecast']



" ------------------- 3. Adapt data to predict future (shift target to predict future values of the target) ------------------- "
# shift data
data[list_target] = data[list_target].shift(-steps_forecast)
data = data.dropna()


" ------------------- ### 4. Split data train/test. Timeseries split ------------------- "
X_train, X_test, y_train, y_test = train_test_split(data[list_features], 
                                                    data[list_target], 
                                                    test_size=0.2, 
                                                    random_state=42,
                                                    shuffle = False
                                                   )


" ------------------- 5. Processing data -moving average ------------------- "
# moving average - the same transformation in train and test data
X_train = X_train.rolling(3).mean().dropna()
X_test = X_test.rolling(3).mean().dropna()

y_train = y_train.rolling(3).mean().dropna()
y_test = y_test.rolling(3).mean().dropna()

print('--- train ---')
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)

print('--- test ---')
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)


" ------------------- 6. Train differents models ------------------- "

################ 6.3 Define parameters of vertex experiment ################
# PARAMETERS TO CREATE AN EXPERIMENT IN VERTEX AI
# obs: In names only are accepted '[a-z0-9][a-z0-9-]{0,127}'
EXPERIMENT_NAME = NAME_DATASET # the name of the vertex experiment is the name of the dataset
EXPERIMENT_DESCRIPTION = f'Run forecasting models of a target. Dataset: {EXPERIMENT_NAME}'


################ 6.4 Set experiment vertex ################
# search tensorboard instance, if it doesn't exist -> created it
id_tensorboard_vertex = get_tensorboard_instance_or_create(experiment_name = EXPERIMENT_NAME,
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



################ 6.5 linear regression (lr) ################
""" RUN NAME IN EXPERIMENT """
RUN_NAME = "run-lr"
print('---- trainning model: ', RUN_NAME)


""" train model """
# create model - train it - evaluate it
lr = LinearRegression() # create model
lr.fit(X_train, y_train) # train
y_test_predicted = lr.predict(X_test) # predict
r2_lr, rmse_lr, mae_lr = evaluate_model(y_test, y_test_predicted) # evaluate metrics
plot_y_true_y_pred, path_y_true_y_pred = plot_y_true_vs_y_pred(y = y_test, y_pred = y_test_predicted, title_plot = f'model: {RUN_NAME}') # Ytrue_vs_Ypred


""" registry run in experiment """
# create a run
vertex_ai.start_run(RUN_NAME)


# define params to save. In a dicctionary
params_problem = {
    'steps_forecast':steps_forecast
}
# save parameters
vertex_ai.log_params(params_problem)


# define metrics to save. In a dicctionary
metrics_to_save = {
    'r2': r2_lr,
    'rmse': rmse_lr,
    'mae': mae_lr
}

# save metrics
vertex_ai.log_metrics(metrics_to_save)

# save graphs
print('saving plot y_true vs y_pred ...')
save_artifacts_experiments_vertex(path_artifact_locally = path_y_true_y_pred, # plot y_true vs y_pred
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save model (but not registry)
print('saving model ...')
model_name = 'model.pkl'
with open(model_name, "wb") as output: # save locally
    pickle.dump(lr, output)
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = model_name,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_train
print('saving X_train ...')
artifact_data = 'X_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_train, output)# change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_train
print('saving y_train ...')
artifact_data = 'y_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_train, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_test
print('saving X_test ...')
artifact_data = 'X_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_test
print('saving y_test ...')
artifact_data = 'y_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

### terminar run
vertex_ai.end_run()



################ 6.6 decision tree (tree) ################
""" RUN NAME IN EXPERIMENT """
RUN_NAME = "run-tree"


""" train model """
# define params to save. In a dicctionary
params_training = {
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 10,
    'random_state': 42
}

# create model - train it - evaluate it
tree = DecisionTreeRegressor(**params_training) # create model
tree.fit(X_train, y_train) # train
y_test_predicted = tree.predict(X_test) # predict
r2_tree, rmse_tree, mae_tree = evaluate_model(y_test, y_test_predicted) # evaluate metrics
plot_y_true_y_pred, path_y_true_y_pred = plot_y_true_vs_y_pred(y = y_test, y_pred = y_test_predicted, title_plot = f'model: {RUN_NAME}') # Ytrue_vs_Ypred


""" registry run in experiment """
# create a run
vertex_ai.start_run(RUN_NAME)

# save parameters
params_problem = {'steps_forecast':steps_forecast}
vertex_ai.log_params(params_training) # parameters of the model trained
vertex_ai.log_params(params_problem) # parameters of the problem that ML model try to solve

# define metrics to save. In a dicctionary
metrics_to_save = {
    'r2': r2_tree,
    'rmse': rmse_tree,
    'mae': mae_tree
}

# save metrics
vertex_ai.log_metrics(metrics_to_save)

# save graphs
print('saving plot y_true vs y_pred ...')
save_artifacts_experiments_vertex(path_artifact_locally = path_y_true_y_pred, # plot y_true vs y_pred
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save model (but not registry)
print('saving model ...')
model_name = 'model.pkl'
with open(model_name, "wb") as output: # save locally
    pickle.dump(tree, output)
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = model_name,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_train
print('saving X_train ...')
artifact_data = 'X_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_train, output)# change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_train
print('saving y_train ...')
artifact_data = 'y_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_train, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_test
print('saving X_test ...')
artifact_data = 'X_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_test
print('saving y_test ...')
artifact_data = 'y_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

### terminar run
vertex_ai.end_run()


################ 6.7 random forest (small) (rf_small) ################
""" RUN NAME IN EXPERIMENT """
RUN_NAME = "run-rf-small"


""" train model """
# define params to save. In a dicctionary
params_training = {
    'n_estimators': 5,
    'max_depth': 50,
    'min_samples_split': 10,
    'min_samples_leaf': 10,
    'random_state': 42
}

# create model - train it - evaluate it
rf_small = RandomForestRegressor(**params_training) # create model
rf_small.fit(X_train, y_train) # train
y_test_predicted = rf_small.predict(X_test) # predict
r2_rf_small, rmse_rf_small, mae_rf_small = evaluate_model(y_test, y_test_predicted) # evaluate metrics
plot_y_true_y_pred, path_y_true_y_pred = plot_y_true_vs_y_pred(y = y_test, y_pred = y_test_predicted, title_plot = f'model: {RUN_NAME}') # Ytrue_vs_Ypred


""" registry run in experiment """
# create a run
vertex_ai.start_run(RUN_NAME)

# save parameters
params_problem = {'steps_forecast':steps_forecast}
vertex_ai.log_params(params_training) # parameters of the model trained
vertex_ai.log_params(params_problem) # parameters of the problem that ML model try to solve

# define metrics to save. In a dicctionary
metrics_to_save = {
    'r2': r2_rf_small,
    'rmse': rmse_rf_small,
    'mae': mae_rf_small
}

# save metrics
vertex_ai.log_metrics(metrics_to_save)

# save graphs
print('saving plot y_true vs y_pred ...')
save_artifacts_experiments_vertex(path_artifact_locally = path_y_true_y_pred, # plot y_true vs y_pred
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save model (but not registry)
print('saving model ...')
model_name = 'model.pkl'
with open(model_name, "wb") as output: # save locally
    pickle.dump(rf_small, output)
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = model_name,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_train
print('saving X_train ...')
artifact_data = 'X_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_train, output)# change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_train
print('saving y_train ...')
artifact_data = 'y_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_train, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_test
print('saving X_test ...')
artifact_data = 'X_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_test
print('saving y_test ...')
artifact_data = 'y_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

### terminar run
vertex_ai.end_run()



################ 6.8 random forest (medium) (rf_medium) ################
""" RUN NAME IN EXPERIMENT """
RUN_NAME = "run-rf-medium"


""" train model """
# define params
params_training = {
    'n_estimators': 30,
    'max_depth': 50,
    'min_samples_split': 10,
    'min_samples_leaf': 10,
    'random_state': 42
}

# create model - train it - evaluate it
rf_medium = RandomForestRegressor(**params_training) # create model
rf_medium.fit(X_train, y_train) # train
y_test_predicted = rf_medium.predict(X_test) # predict
r2_rf_medium, rmse_rf_medium, mae_rf_medium = evaluate_model(y_test, y_test_predicted) # evaluate metrics
plot_y_true_y_pred, path_y_true_y_pred = plot_y_true_vs_y_pred(y = y_test, y_pred = y_test_predicted, title_plot = f'model: {RUN_NAME}') # Ytrue_vs_Ypred


""" registry run in experiment """
# create a run
vertex_ai.start_run(RUN_NAME)

# save parameters
params_problem = {'steps_forecast':steps_forecast}
vertex_ai.log_params(params_training) # parameters of the model trained
vertex_ai.log_params(params_problem) # parameters of the problem that ML model try to solve

# define metrics to save. In a dicctionary
metrics_to_save = {
    'r2': r2_rf_medium,
    'rmse': rmse_rf_medium,
    'mae': mae_rf_medium
}

# save metrics
vertex_ai.log_metrics(metrics_to_save)

# save graphs
print('saving plot y_true vs y_pred ...')
save_artifacts_experiments_vertex(path_artifact_locally = path_y_true_y_pred, # plot y_true vs y_pred
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save model (but not registry)
print('saving model ...')
model_name = 'model.pkl'
with open(model_name, "wb") as output: # save locally
    pickle.dump(rf_medium, output)
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = model_name,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_train
print('saving X_train ...')
artifact_data = 'X_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_train, output)# change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_train
print('saving y_train ...')
artifact_data = 'y_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_train, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_test
print('saving X_test ...')
artifact_data = 'X_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_test
print('saving y_test ...')
artifact_data = 'y_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

### terminar run
vertex_ai.end_run()




################ 6.9 random forest (default) (rf_default) ################
""" RUN NAME IN EXPERIMENT """
RUN_NAME = "run-rf-default"


""" train model """
# define params
### TODO: SEE THAT PARAMS INPUT IN THE MODEL ARE USED TO REGISTRY IN VERTEX EXPERIMENTS
params_training = {
    'n_estimators': 100,
    'max_depth': 50,
    'min_samples_split': 10,
    'min_samples_leaf': 10,
    'random_state': 42
}

# create model - train it - evaluate it
rf_default = RandomForestRegressor(**params_training) # create model
rf_default.fit(X_train, y_train) # train
y_test_predicted = rf_default.predict(X_test) # predict
r2_rf_default, rmse_rf_default, mae_rf_default = evaluate_model(y_test, y_test_predicted) # evaluate metrics
plot_y_true_y_pred, path_y_true_y_pred = plot_y_true_vs_y_pred(y = y_test, y_pred = y_test_predicted, title_plot = f'model: {RUN_NAME}') # Ytrue_vs_Ypred


""" registry run in experiment """
# create a run
vertex_ai.start_run(RUN_NAME)

# save parameters
params_problem = {'steps_forecast':steps_forecast}
vertex_ai.log_params(params_training) # parameters of the model trained
vertex_ai.log_params(params_problem) # parameters of the problem that ML model try to solve

# define metrics to save. In a dicctionary
metrics_to_save = {
    'r2': r2_rf_default,
    'rmse': rmse_rf_default,
    'mae': mae_rf_default
}

# save metrics
vertex_ai.log_metrics(metrics_to_save)

# save graphs
print('saving plot y_true vs y_pred ...')
save_artifacts_experiments_vertex(path_artifact_locally = path_y_true_y_pred, # plot y_true vs y_pred
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save model (but not registry)
print('saving model ...')
model_name = 'model.pkl'
with open(model_name, "wb") as output: # save locally
    pickle.dump(rf_default, output)
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = model_name,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_train
print('saving X_train ...')
artifact_data = 'X_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_train, output)# change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_train
print('saving y_train ...')
artifact_data = 'y_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_train, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_test
print('saving X_test ...')
artifact_data = 'X_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_test
print('saving y_test ...')
artifact_data = 'y_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

### terminar run
vertex_ai.end_run()




################ 6.10 NN MLP (mlp-sk) ################
""" RUN NAME IN EXPERIMENT """
RUN_NAME = "run-mlp-sk"


""" train model """
# define params to save. In a dicctionary
params_training = {
    'hidden_layer_sizes': '[200, 100, 50, 25]',  # only accepted float, integer or string
    'activation': 'relu',
    'learning_rate_init': 0.001,
    'max_iter': 200,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'random_state': 42
}
params_to_train = dict(list(params_training.items())[1:])  #define new params becuase vertex doesn't accept a list

### parameters
hidden_layer_sizes_nn_mlp = [200, 100, 50, 25]

# create model - train it - evaluate it
nn_mlp = MLPRegressor(hidden_layer_sizes = [200, 100, 50, 25], **params_to_train) # create model
nn_mlp.fit(X_train, y_train) # train
y_test_predicted = nn_mlp.predict(X_test) # predict
r2_nn_mlp, rmse_nn_mlp, mae_nn_mlp = evaluate_model(y_test, y_test_predicted) # evaluate metrics
plot_y_true_y_pred, path_y_true_y_pred = plot_y_true_vs_y_pred(y = y_test, y_pred = y_test_predicted, title_plot = f'model: {RUN_NAME}') # Ytrue_vs_Ypred


""" registry run in experiment """
# create a run
vertex_ai.start_run(RUN_NAME)

# save parameters
params_problem = {'steps_forecast':steps_forecast}
vertex_ai.log_params(params_training) # parameters of the model trained
vertex_ai.log_params(params_problem) # parameters of the problem that ML model try to solve

# define metrics to save. In a dicctionary
metrics_to_save = {
    'r2': r2_nn_mlp,
    'rmse': rmse_nn_mlp,
    'mae': mae_nn_mlp
}

# save metrics
vertex_ai.log_metrics(metrics_to_save)

# save graphs
print('saving plot y_true vs y_pred ...')
save_artifacts_experiments_vertex(path_artifact_locally = path_y_true_y_pred, # plot y_true vs y_pred
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save model (but not registry)
print('saving model ...')
model_name = 'model.pkl'
with open(model_name, "wb") as output: # save locally
    pickle.dump(nn_mlp, output)
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = model_name,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_train
print('saving X_train ...')
artifact_data = 'X_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_train, output)# change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_train
print('saving y_train ...')
artifact_data = 'y_train.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_train, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X_test
print('saving X_test ...')
artifact_data = 'X_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y_test
print('saving y_test ...')
artifact_data = 'y_test.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y_test, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_GCP, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

### terminar run
vertex_ai.end_run()






" ------------------- Guardar modelo entrenado para ser registrado en vertex models - menu models - GUARDAR REGRESIÓN LINEAL POR DEFECTO - no retornar error ------------------- "
# DEFINIR PATH CUSTOM DONDE SE VA A GUARDAR EL MODELO. Para registrar modelo en vertex obligatoriamente debe existir el path ".../model/model.pkl"
# En los códigos posteriores se crea el path completo. Aquí se crea hasta el folder ".../model/"
# En el código que envia el job de entrenamiento se debe especificar el mismo path para decir que ahí se guardará el artefacto del modelo
path_artifact_model_vertex = f'gs://{BUCKET_GCP}/vertex-ai-registry-model/{NAME_DATASET}/run_{date_time}/model/'
print('path del modelo a GCS: ', path_artifact_model_vertex)


# Save model artifact to local filesystem (doesn't persist)
artifact_filename = 'model.pkl'
local_path = artifact_filename
with open(local_path, 'wb') as model_file:
    pickle.dump(lr, model_file)
print('modelo guardado local')


# Upload model artifact to Cloud Storage - try: guardar en path de GCS definido // except: error al guardar
try:
    model_directory = path_artifact_model_vertex
    storage_path = os.path.join(model_directory, artifact_filename) # generar path completo "gs//.../model/model.pkl"
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(local_path)
    print('MODELO GUARDADO EN GCS')
except Exception as e: 
    print('Error: ', str(e))
    print('MODELO NO GUARDADO EN GCS')


# delete model artifact saved locally (save locally in a job don't save permanently the file)
os.remove(artifact_filename)