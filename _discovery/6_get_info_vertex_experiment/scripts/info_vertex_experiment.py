from google.cloud import aiplatform as vertex_ai
import pandas as pd
import gcsfs
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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



def show_information_run(SELECTED_RUN, EXPERIMENT_NAME):
    """
    Connect to vertex experiment and show the information of training of one model

    Given a certain experiment (name of dataset uploaded to the app) and given a certain run (name of the model trained and saved). Connect and show
    the information (params, metrics, artifacts)
    """

    ###### connect to the run in vertex
    run_to_show = vertex_ai.ExperimentRun.get(SELECTED_RUN, experiment = EXPERIMENT_NAME)
    
    ###### get params values. filter only the parameters saved
    params_trainning = run_to_show.get_params()
    params_trainning = pd.DataFrame([params_trainning])
    
    # divide params training in 2 groups. Params of the model trained. Params of the problem
    params_problem = params_trainning[['steps_forecast']]
    params_trainning = params_trainning.drop(columns = ['steps_forecast'])
    
    ###### get metrics values. filter only the metrics saved
    metrics_trainning = run_to_show.get_metrics()
    metrics_trainning = pd.DataFrame([metrics_trainning])
    
    ###### show artifact plot y_true vs y_pred
    
    # get list artifacts
    list_paths_artifacts = run_to_show.get_artifacts()
    
    # get path plot y_true vs y_pred in GCS. All the artifacts saved are .pkl except the plot (.png). Search the artifact that are a .png image
    for index_artifact in range(len(list_paths_artifacts)):
        if list_paths_artifacts[index_artifact].uri.split('.')[-1] == 'png':
            path_plt_plot = list_paths_artifacts[index_artifact].uri
        else:
            pass
    
    # connect to GCS as pythonic way
    fs = gcsfs.GCSFileSystem()
    
    # read the PNG file since GCS
    with fs.open(path_plt_plot, 'rb') as file:
        img_true_pred = mpimg.imread(file)
    
    return params_trainning, params_problem, metrics_trainning, img_true_pred