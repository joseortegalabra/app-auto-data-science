import pandas as pd
import numpy as np
from google.cloud import bigquery
import gcsfs
import pickle

# metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import iqr

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# explanaible AI
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence

# #shap - shapash
# import shap
# from shapash import SmartExplainer





# ------------------------------------------------------------- Load data train and test ------------------------------------------------------------- #
def load_data(bucket_gcp, name_dataset, selected_run):
    """
    Load data used in the model
    """
    # X_train
    path_X_train = f'gs://{bucket_gcp}/{name_dataset}/{selected_run}/X_train.pkl'
    X_train = pd.read_pickle(path_X_train)
    
    # y_train
    path_y_train = f'gs://{bucket_gcp}/{name_dataset}/{selected_run}/y_train.pkl'
    y_train = pd.read_pickle(path_y_train)
    
    # X_test
    path_X_test = f'gs://{bucket_gcp}/{name_dataset}/{selected_run}/X_test.pkl'
    X_test = pd.read_pickle(path_X_test)
    
    # y_test
    path_y_test = f'gs://{bucket_gcp}/{name_dataset}/{selected_run}/y_test.pkl'
    y_test = pd.read_pickle(path_y_test)
    
    print('SHAPE DATA')
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    
    print('\n')
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)

    return X_train, y_train, X_test, y_test




# ------------------------------------------------------------- 0. Get predictions ------------------------------------------------------------- #
def get_predictions(model, X, y):
    """
    get predictions
    """
    # predict
    y_pred = model.predict(X)
    
    # transform dataframe
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = y.columns
    y_pred.index = y.index
    
    return y_pred



# ------------------------------------------------------------- 1. Metrics ------------------------------------------------------------- #
#### 1.1 Metric for Model trained
def calculate_metrics_regressors_models(y, y_pred, model_name, decimals_round = None):
    """
    Calculate a certain number of metrics to evaluate regression models. The metrics are rounded to X decimals

    Args
        y (dataframe): y true
        y_pred (dataframe): y predicted with the model. In this codes are passed y_pred instead of X
        model_name (string): name of the model. This name is used when the metrics are saved to identify the model of these metrics
        decimals_round = Number of decimals to round the values. Defult None, no round the values.

    Return
        metrics_regressors (dataframe): dataframe with the metrics of the model in this datasets. Row: name metrics. Columns: value metrics
    """

    #### R2
    r2 = r2_score(y, y_pred)
    
    #### MSE
    mse = mean_squared_error(y, y_pred, squared = True)
    
    #### RMSE
    rmse = mean_squared_error(y, y_pred, squared = False)
    
    #### RMSE_MEAN_RATIO
    # rmse mean ratio: rmse / mean_y_true
    rmse_mean_ratio = rmse / y.mean().values[0]
    rmse_mean_ratio = round(100 * (rmse_mean_ratio), 2)
    
    #### RMSE_IQR_RATIO
    # rmse iqr ratio: rmse / iqr_y_true
    rmse_iqr_ratio = rmse / iqr(y)
    rmse_iqr_ratio = round(100 * (rmse_iqr_ratio), 2)
    
    #### MAE
    mae = mean_absolute_error(y, y_pred)
    
    #### MAE_RATIO
    mae_mean_ratio = mae / y.mean().values[0]
    mae_mean_ratio = round(100 * (mae_mean_ratio), 2)
    
    #### MAE_IQR_RATIO
    mae_iqr_ratio = mae / iqr(y)
    mae_iqr_ratio = round(100 * (mae_iqr_ratio), 2)
    
    
    
    #### JOIN INTO ONE DATAFRAME
    # create dataframe
    metrics_regressors = pd.DataFrame(index = [model_name])
    
    # add metrics
    metrics_regressors['r2'] = r2
    metrics_regressors['mse'] = mse
    metrics_regressors['rmse'] = rmse
    metrics_regressors['rmse_mean_ratio(%)'] = rmse_mean_ratio
    metrics_regressors['rmse_iqr_ratio(%)'] = rmse_iqr_ratio
    metrics_regressors['mae'] = mae
    metrics_regressors['mae_mean_ratio(%)'] = mae_mean_ratio
    metrics_regressors['mae_iqr_ratio(%)'] = mae_iqr_ratio
    
    # round
    metrics_regressors = metrics_regressors.astype('float')
    if decimals_round:
        metrics_regressors = metrics_regressors.round(decimals_round)


    return metrics_regressors



# ------------------------------------------------------------- 2. Plot Predictions ------------------------------------------------------------- #
#### 2.1 y_true vs y_pred
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

    plt.close()
    return fig


#### 2.2 hist errors
def hist_errors_predictions(y, y_pred, title_plot, n_bins = 10):
    """
    Plot histogram of error in prediction: errors: abs(y_true vs y_pred) (using matplotlib figure)

    Args:
        y (dataframe): dataframe with y-true values 
        y_pred (dataframe): dataframe with y-pred values
        title_plot (string): tittle in the plot
        n_bins (integer): number of bins in the histogram. Default = 10
    
    Return
        fig (figure matplolib): figure to show, download, etc
    """
    # calculate error
    errors = y - y_pred
    errors = np.abs(errors) # error in abs value
    
    # hist error
    fig = plt.figure()
    plt.hist(errors, bins = n_bins)
    plt.xlabel('Error')
    plt.ylabel('Freq')
    plt.title(f'Histogram of Errors in Predictions:  abs(y - y_pred) - {title_plot}')

    plt.close()
    return fig



# ------------------------------------------------------------- 3. Explanaible AI ------------------------------------------------------------- #
#### 3.1 Permutation Importances
def permutation_importances(model, list_features, X, y):
    """
    Calculate permutation importances
    """
    # calculate permutation importances
    results = permutation_importance(estimator = model, 
                                     X = X, 
                                     y = y, 
                                     n_repeats = 30,
                                     random_state = 42
                                    )

    # define a series with importances (mean) of each feature
    df_importances = pd.Series(results.importances_mean, index = list_features)

    # plot
    fig, ax = plt.subplots()
    df_importances.plot.bar(yerr = results.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation")
    ax.set_ylabel("Mean R2 decrease")
    fig.tight_layout()

    plt.close()
    return fig, df_importances


#### 3.2 Permutation Importances with Noise Features
def permutation_importances_noise(model, X_train, y_train, X_test, y_test):
    """
    Calculate permutation importances adding noise in the dataset
    """
    
    # create instance model and data noise since original instance model trained
    import copy
    model_noise = copy.deepcopy(model)
    
    X_train_noise = X_train.copy()
    y_train_noise = y_train.copy()
    X_test_noise = X_test.copy()
    y_test_noise = y_test.copy()

    # create dataframe train - test with noise
    np.random.seed(42)
    X_train_noise['noise_1'] = np.random.normal(size = X_train.shape[0])
    X_train_noise['noise_2'] = 10 * np.random.normal(size = X_train.shape[0])
    
    X_test_noise['noise_1'] = np.random.normal(size = X_test.shape[0])
    X_test_noise['noise_2'] = 10 * np.random.normal(size = X_test.shape[0])

    # train model
    model_noise.fit(X_train_noise, y_train_noise)
    
    # get predictions
    y_test_noise_pred = model_noise.predict(X_test_noise)
    
    #### PDP - utilizar funcion anterior - re utilizar funcion
    # calculate permutation importances
    results = permutation_importance(estimator = model_noise, 
                                     X = X_test_noise, 
                                     y = y_test_noise, 
                                     n_repeats = 30,
                                     random_state = 42
                                    )
    
    # define a series with importances (mean) of each feature
    df_importances = pd.Series(results.importances_mean, index = X_test_noise.columns.tolist() )
    
    # plot
    fig, ax = plt.subplots()
    df_importances.plot.bar(yerr = results.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation")
    ax.set_ylabel("Mean R2 decrease")
    fig.tight_layout()
    
    plt.close()

    return fig


#### 3.3 Partial Dependence Plot
def pdp_plot_old(model, X):
    """
    Partial depedence plot
    """
    
    # plot
    display = PartialDependenceDisplay.from_estimator(
        estimator = model,
        X = X,
        random_state = 42,
        features = X.columns.tolist(),
        n_cols = 3
    )
    
    # Crear una figura con el tamaño deseado
    fig = plt.figure(figsize=(15, 10))
    
    # Mostrar el gráfico
    display.plot(ax=plt.gca())  # Utiliza el eje actual de la figura actual
    plt.tight_layout()

    plt.close()
    return fig

def pdp_plot(model, X):
    """
    Partial depedence plot
    """
    
    # plot
    display = PartialDependenceDisplay.from_estimator(
        estimator = model,
        X = X,
        random_state = 42,
        features = X.columns.tolist(),
        n_cols = 3
    )

    # get plot
    fig_pdp_plot = plt.gcf() # get actual plt figure
    fig_pdp_plot.set_size_inches(15, 10) # change size plot
    plt.close() # close plt

    return fig_pdp_plot


# ------------------------------------------------------------- 4. Perturbation test - Sensitivy Analysis ------------------------------------------------------------- #
#### 4.1 Peturbation test one feature global plots
def perturbation_test_one_feature_global_analysis(tag_sensitivy_analysis, epsilon, model, X, y, y_pred):
    """
    (Perturbation +- epsilon) in one feature and predict with this perturbations. Plot a histogram of predited values with
    "true values", "pred values", "pred values - epsilon" and "pred values + epsilon"

    Args
        tag_sensitivy_analysis (string): tag sensitivy analysis
        epsilon (float/integer): epsilon perturbation in the data
        X (dataframe): features dataframe
        y (dataframe): target dataframe (true values)
        y_pred (dataframe): target predicted dataframe (prediction without perturbation)

    Return
        plot
    """
    
    ###### get list of original features
    list_original_features = X.columns.tolist()
    
    ###### clone data
    X_sensitivy = X.copy()
    y_pred_sensitivy = y_pred.copy()
    y_pred_sensitivy.columns = ['target']
    
    ###### calculate the percentual variation of the epsilon in the feature (mean value). how is the percentual variation of the data
    epsilon_percent_impact = round(100 * epsilon / X_sensitivy[tag_sensitivy_analysis].mean(), 2)
    print(f'-- Epsilon percent impact: {epsilon_percent_impact}%')
    
    ###### generate two columns of the feature: (feature-epsilon, feature+epsilon)
    X_sensitivy[tag_sensitivy_analysis + '_-_epsilon'] = X_sensitivy[tag_sensitivy_analysis] - epsilon
    X_sensitivy[tag_sensitivy_analysis + '_+_epsilon'] = X_sensitivy[tag_sensitivy_analysis] + epsilon
    
    
    ###### get the predicted values with the perturbation (feature-epsilon, feature+epsilon)
    # clone list original features to reeplace the feature with epsilon values
    list_features_minus_epsilon = list_original_features.copy()
    list_features_plus_epsilon = list_original_features.copy()
    
    # get the position of the feature with sensitivy analysis
    idx_tag_sensitivy_analysis = list_features_minus_epsilon.index(tag_sensitivy_analysis)
    
    # redefine list of features with the names of features with epsilon values
    list_features_minus_epsilon[idx_tag_sensitivy_analysis] = tag_sensitivy_analysis + '_-_epsilon'
    list_features_plus_epsilon[idx_tag_sensitivy_analysis] = tag_sensitivy_analysis + '_+_epsilon'
    
    ###### model.predict with its epsilon values
    # generate data minus and plus
    data_sensitivy_minus = X_sensitivy[list_features_minus_epsilon] # filter
    data_sensitivy_minus.columns = list_original_features # rename columns
    data_sensitivy_plus = X_sensitivy[list_features_plus_epsilon]
    data_sensitivy_plus.columns = list_original_features
    
    # save predict values
    y_pred_sensitivy['target_'+ '_-_epsilon'] = model.predict(data_sensitivy_minus) # predecir con delta minus de variable de 
    y_pred_sensitivy['target_'+ '_+_epsilon'] = model.predict(data_sensitivy_plus)
    
    
    sns.kdeplot(y, label = 'true')
    sns.kdeplot(y_pred_sensitivy['target'], label = 'pred')
    sns.kdeplot(y_pred_sensitivy['target_'+ '_-_epsilon'], label = 'pred-epsilon(feature)')
    sns.kdeplot(y_pred_sensitivy['target_'+ '_+_epsilon'], label = 'pred+epsilon(feature)')
    plt.legend()

    # get a python variable with plt figure
    fig_plot_sensibility = plt.gcf() # get actual plt figure
    plt.close() # close plt

    return fig_plot_sensibility


# ------------------------------------------------------------- 5. Metrics by segments ------------------------------------------------------------- #
#### 5.1 Segment by most important feature
def metrics_segmentation_analysis(tag_segment, X, y, y_pred, metrics):
    """
    """
    
    # get percentile and bins
    percentile_index, bins = pd.qcut(X[tag_segment], 5, labels=False, retbins=True)

    # add columns percentile in y test
    y_percentile = y.copy()
    y_percentile['percentile_index'] = percentile_index
    
    # add columns percentile in y test pred
    y_pred_percentile = y_pred.copy()
    y_pred_percentile['percentile_index'] = percentile_index


    # show metrics
    metrics_percentile = pd.DataFrame()
    for index in range(5):
        
        # calculate values start and end
        start_segment = round(bins[index], 3)
        end_segment = round(bins[index+1], 3)
    
        # filter test data ground truth
        y_percentile_aux = y_percentile[y_percentile['percentile_index'] == index]
        y_percentile_aux = y_percentile_aux.drop(columns = 'percentile_index')
          
        # filter test data pred
        y_pred_percentile_aux = y_pred_percentile[y_pred_percentile['percentile_index'] == index]
        y_pred_percentile_aux = y_pred_percentile_aux.drop(columns = 'percentile_index')
            
        # calculate metrics
        metrics_percentile_aux = calculate_metrics_regressors_models(y = y_percentile_aux,
                                            y_pred = y_pred_percentile_aux, 
                                            model_name = f'Percentile {index}', 
                                            decimals_round = None
                                           )
    
        # append
        metrics_percentile = pd.concat([metrics_percentile, metrics_percentile_aux])


    # append original score
    output_metrics = pd.concat([metrics, metrics_percentile])


    return output_metrics


# ------------------------------------------------------------- 1. ------------------------------------------------------------- #


# ------------------------------------------------------------- 1. ------------------------------------------------------------- #



# ------------------------------------------------------------- 1. ------------------------------------------------------------- #