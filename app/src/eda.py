import pandas as pd
import numpy as np
import gcsfs
import json

# plotly
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# seaborn
import matplotlib.pyplot as plt
import seaborn as sns


##############################################  3.1 statistics ##############################################
def generate_descriptive_statistics(df):
    """
    Generate descriptive statistics of a dataframe. All the values are rounded by 3 decimals. Generate a dataframe and transform it into a plotly table
    
    Args
        df (dataframe): dataframe input

    Return
        statistics (dataframe): dataframe with statistics
    """
    # generate table to save
    list_percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    statistics = df.describe(percentiles = list_percentiles)
    
    # round 3 decimals
    statistics = statistics.round(3)

    # reset index
    #statistics.reset_index(inplace = True)

    return statistics



##############################################  3.2 Histograms ##############################################
def plot_kde_hist(df, number_columns = 2):
    """
    Plot the histogram and the KDE.
    Using seaborn

    Args
        df (dataframe): data. The index should be the timestamp

    Return
        fig (figure matplotlib): fig of matplotlib with the plot generated
    """

    ############################################################################
    # get list of features
    list_features = df.columns.tolist()
    
    
    # define number of rows with a number of columns fixed pass as parameter
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)

    
    # create subplots
    fig, axes = plt.subplots(nrows = number_rows, 
                             ncols = number_columns,
                             #figsize = (subplot_width * number_columns, subplot_height * number_rows),
                             figsize=(7*number_columns, 4*number_rows + 0),
                             tight_layout = True
                            )
    sns.set(style = "darkgrid", palette="gray")
    
    
    # add title
    #fig.suptitle("Histogram with kde", fontsize=28)  # sometimes the tittle is overlaped in the plots
    
    # add subplot for each of the features -> feature
    for index_feature, feature in enumerate(list_features):
        row = (index_feature // number_columns) #+ 1 # in matplotlib index starts in 0, in plolty starts in 1
        column = (index_feature % number_columns) #+ 1
    
        # subplot each feature
        sns.histplot(df, x = feature, kde=True, color='gray', element='step', fill=True, ax=axes[row, column])
        axes[row, column].set_title(f'Histogram and KDE of "{feature}"')
    
    # adjust design
    plt.subplots_adjust(top=0.95) # sup title above the subplots
    
    ############################## 
    plt.close()
    return fig




##############################################  3.3 Boxplots monthly ##############################################
def plot_multiple_boxplot_months(df, number_columns = 1):
    """
    Plot boxplots of each month and each year. See the montly distribution of ALL features

    Args
        df (datafame): dataframe input
        number_columns (integer): number of columns, by default ONE column

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # get list of features
    list_features = df.columns.tolist()

    # get number of rows (number row = number of data / number of columns)
    # (considering fixed the number of columns) 
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)


    ############################## 
    # create subplots
    fig = make_subplots(rows = number_rows, 
                        cols = number_columns, 
                        subplot_titles = df.columns,
                        shared_xaxes=False,
                        vertical_spacing = 0.015
                       )

    # add subplot of boxplots for each month and year
    for index_feature, feature in enumerate(list_features):

        # obtener índices en el subplot (en plotly los índices comienzan en 1, por lo que debe sumarse un 1 a los resultados obtenidos)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1
        
        # boxplot
        box_fig = px.box(df, x=df.index.month, y=feature, color=df.index.year)
        for trace in box_fig.data:
            fig.add_trace(trace, row = row, col = column)


  # adjust plot
    fig.update_layout(title = 'Boxplots for Month and Year',
                      xaxis_title='Month',
                      yaxis_title='Value',
                      legend_title='Year',
                      title_x=0.5,  # center
                      title_font=dict(size=20),
                      height = 550 * number_rows,  # largo 650
                      width = 1050 * number_columns, # ancho 1850
                      showlegend=True,
                      boxmode='group',  # Group boxplots by month
                      boxgap=0.2)  # Adjust the gap between grouped boxplots
    ############################## 

    return fig



##############################################  3.4 Correlations - all ##############################################
def calculate_correlations_triu(df):
    """
    Given a dataframe, calculate the correlations (pearson) between all the variables in the dataframe
    Args
        df (dataframe)

    Return
        df_corr (dataframe): dataframe with correlations
        df_corr_upper(dataframe): dataframe with correltions - upper triangular matrix - round by 2 decimals
  """

    # calculate correlations
    df_corr = df.corr(method='pearson')
    
    # upper triangular matrix
    df_corr_upper = df_corr.where(np.triu(np.ones(df_corr.shape)).astype('bool'))
    
    # round 2 decimals
    df_corr = np.round(df_corr, 2)
    df_corr_upper = np.round(df_corr_upper, 2)
    
    return df_corr, df_corr_upper

def plot_heatmap(df_corr):
    """
    Plot heatmap using the input dataframe
    It could be used to plot the correlations between differents variables

    Args
        df_corr (dataframe): dataframe with correlations to plot

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # heatmap
    fig = px.imshow(df_corr, text_auto=True, aspect="auto")
    
    # change title
    fig.update_layout(
      title_text = "Correlations",
        title_x = 0.5,
    title_font = dict(size = 28)
      )
    
    return fig




##############################################  3.5 Correlations - target ##############################################
def calculate_correlations_target(df, target):
    """
    Given a dataframe and a target (that will be present in the dataframe) calculate the correlations of all features agains the target

    Args
        df (dataframe): dataframe
        target (string): feature target - that will be present in the dataframe
    
    Return
        df_corr (dataframe): dataframe with the correlations
    """

    # calculate correlations select only with the target
    df_corr_target = df.corr(method='pearson')[[target]]
    
    # roudn 3 decimals
    df_corr_target = np.round(df_corr_target, 3)
    
    # transpose to see in a better way
    df_corr_target = df_corr_target.T
    
    return df_corr_target




##############################################  3.6 Segmentation ##############################################
def custom_segmentation(df, var_segment, intervals_segments, labels_segments):
    """
    Given a dataframe, generate a new column with a categorical values that divide the data in differents segments. 
    Segment the data by a certain variable with a custom segmentation
    
    Args
        df (dataframe): dataframe input
        var_segment (string): variable feature/target used to segment the data
        intervals_segments (list of numbers): list with the thresholds used to segment the data
        labels_segments (list of strings): list with the names of the differents segments generated. Shape: len(intervals_segments) - 1

    Return
        df(dataframe): the input dataframe with a new column with the segment
    """

    # apply pd.cut to generate intervals
    df[f'{var_segment}_segments'] = pd.cut(df[var_segment], 
                                           bins = intervals_segments, 
                                           labels = labels_segments, 
                                           include_lowest = True
                                          )

    # order data by the custom segmentation - to generate plots it is neccesary to sort the data
    # if the plot show a temporal relation like trends plots, it is necessary sort the data by index
    df = df.sort_values(by = [var_segment])
    
    return df




##############################################  3.7 Segmentation - freq ##############################################
def plot_freq_segmentation(df, var_segment):
    """
    Given a segmentation in the data, plot the freq of each segment
    
    Args
        df (dataframe): input dataframe
        var_segment (string): variable in the input dataframe that indicate the segments in the data
    
    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ''' calculate dataframe with freq '''
    df_freq_segmentation = df[var_segment].value_counts()
    df_freq_segmentation = pd.DataFrame(df_freq_segmentation)
    df_freq_segmentation.reset_index(inplace = True)
    
    
    ''' plot barplot freq '''
    # create freq bar
    fig = px.histogram(df_freq_segmentation, x = var_segment, y = 'count', barmode='group')
    
    # add value each bar
    fig.update_traces(text = df_freq_segmentation['count'], textposition='outside')
    
    # update layout
    fig.update_layout(
        title_text=f'Freq of each segments for segmentation by {var_segment}',
        title_x=0.5,  # centrar título
        title_font=dict(size=20),
        yaxis=dict(title = 'freq')
    )

    return fig



##############################################  3.8 Segmentation - boxplot ##############################################
def plot_boxplots_segments(df, var_segment, number_columns = 2):
    """
    Plot multiple boxplots for each feature in the dataframe. Differents colors in the histogram according the segmentation in the data
    
    Args
        df (datafame): input dataframe
        varg_segment (string): name of the column in the input dataframe that indicate the differents segments in the data

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # get list features
    list_features = df.columns.tolist()


    # get number of rows (number row = number of data / number of columns)
    # (considering fixed the number of columns) 
    if (df.shape[1] % number_columns) != 0:
        number_rows = (df.shape[1] // number_columns) + 1 
    else:
        number_rows = (df.shape[1] // number_columns)


    ############################## 
    # Create los subplots
    fig = make_subplots(rows = number_rows, cols = number_columns, shared_xaxes=False, subplot_titles=list_features, 
                        vertical_spacing = 0.03)

    # add each boxplot
    for index_feature, feature in enumerate(list_features):

        # obtener índices en el subplot (en plotly los índices comienzan en 1, por lo que debe sumarse un 1 a los resultados obtenidos)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1
        
        # add trace boxplot
        #fig.add_trace(go.Box(x=df[var_segment], y=df[feature], name = f'Boxplot {feature} by segments {var_segment}'),
        fig.add_trace(go.Box(x=df[var_segment], y=df[feature]),
                row = row,
                col = column)
        
    # update layout
    fig.update_layout(height=len(list_features)*200, 
                      width= 1050, # 1600 original 
                      title_text = "Boxplots Segmentations",
                      title_x = 0.5, # centrar titulo
                    title_font = dict(size = 28)
                     )
    return fig


##############################################  3.9 Segmentation - corr - target ##############################################
def calculate_correlations_triu_segmentation(df, var_segment):
    """
    Given a dataframe and a variable that are segmented the data calculate the correlations (pearson) between all the variables
    Args
        df (dataframe): input dataframe
        var_segment (string): variable in the input dataframe that indicate the segments in the data

    Return
        dict_df_corr_segment(dict): dictionary of dataframes with correltions for each segment - upper triangular matrix - round by 3 decimals
  """

    # get name of each segment in a list
    unique_values_segments = df[var_segment].unique().tolist()
    unique_values_segments = list(filter(pd.notna, unique_values_segments)) # delete null values in segments
    unique_values_segments.sort()

    # generate a list of dataframes with each dataframe is the df_corr for each segment
    dict_df_corr_segment = {}
    for name_segment in unique_values_segments:
    
        # generate auxiliar df for each segment
        df_aux = df[df[var_segment] == name_segment]
        df_aux = df_aux.drop(columns = var_segment)
    
        # calculate corr triu with 3 decimals
        df_corr_segment_aux = df_aux.corr()
        df_corr_segment_aux_upper = df_corr_segment_aux.where(np.triu(np.ones(df_corr_segment_aux.shape), k=1).astype('bool'))
        df_corr_segment_aux_upper = np.round(df_corr_segment_aux_upper, 3)
    
        # append to list
        dict_df_corr_segment[name_segment] = df_corr_segment_aux_upper
    
    return dict_df_corr_segment

def plot_corr_segmentation_subplots_heatmap(dict_df_corr_segment, number_columns = 1):
    """
    Given a dictionary with the correlations for each segment, plot it into a format a subplots of heatmaps

    Args
        dict_df_corr_segment (dict): dictionary where each element is a dataframe with the correlations for each segment
        number_columns (int): for the dimensions of heatmaps set it always in 1

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """
    
    # get list of segments - keys in the dict
    list_segments = list(dict_df_corr_segment.keys())
    
    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_segments) % number_columns) != 0:
        number_rows = (len(list_segments) // number_columns) + 1
    else:
        number_rows = (len(list_segments) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, subplot_titles=tuple(list_segments),
                       vertical_spacing = 0.08)

    ########## for each feature plot:
    for index_segment in range(len(list_segments)):
        segment = list_segments[index_segment]

        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_segment // number_columns) + 1
        column = (index_segment % number_columns) + 1


        # get fig individual
        fig_aux = px.imshow(dict_df_corr_segment[segment], text_auto=True, aspect="auto")
        
        # add scatter to fig global
        fig.add_trace(fig_aux.data[0],
            row = row,
            col = column
        )
    
    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = "Correlations features for each segment",
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig



##############################################  3.10 Segmentation - corr - target ##############################################
def calculate_correlations_target_segmentation(df, var_segment, target):
    """
    Given a dataframe and a variable that are segmented the data calculate the correlations (pearson) between all the features against the target
    Args
        df (dataframe): input dataframe
        var_segment (string): variable in the input dataframe that indicate the segments in the data
        target (string): target

    Return
        dict_df_corr_segment(dict): dictionary of dataframes with correltions for each segment - upper triangular matrix - round by 3 decimals
  """

    # get name of each segment in a list
    unique_values_segments = df[var_segment].unique().tolist()
    unique_values_segments = list(filter(pd.notna, unique_values_segments)) # delete null values in segments
    unique_values_segments.sort()

    # generate a list of dataframes with each dataframe is the df_corr for each segment
    dict_df_corr_segment = {}
    for name_segment in unique_values_segments:
    
        # generate auxiliar df for each segment
        df_aux = df[df[var_segment] == name_segment]
        df_aux = df_aux.drop(columns = var_segment)
    
        # calculate corr triu with 3 decimals
        df_corr_segment_aux = df_aux.corr()[[target]]
        df_corr_segment_aux = np.round(df_corr_segment_aux, 3)
        df_corr_segment_aux = df_corr_segment_aux.T
    
        # append to list
        dict_df_corr_segment[name_segment] = df_corr_segment_aux
    
    return dict_df_corr_segment


##############################################  3.11 Categorical Analysis ##############################################
def generate_labels_percentile_segmentation(df, var_segment, list_percentile, list_labels_percentile_base):
    """
    Given a dataframe and a feature to segment in percentiles, calculate the labels of the segmentation
    
    Choices of labels:
        labels_percentile: ['q1', 'q2', 'q3', 'q4']
        labels_values: ['(0.15-1.2)', '(1.2-1.8)', '(1.8-2.65)', '(2.65-5.0)']
        labels_percentile_values: ['q1 - (0.15-1.2)', 'q2 - (1.2-1.8)', 'q3 - (1.8-2.65)', 'q4 - (2.65-5.0)']
        
    Args
        df (dataframe): dataframe input
        var_segment (string): variable feature/target used to segment the data
        list_percentile (list): list of floats with the percentiles to divide the data
        list_labels_percentile_base (list): list of strings with the base labels of percentiles to divide the data 

    Return
        list_labels_percentile_base, list_labels_values_range, list_labels_percentile_values_range (lists). list of the 3 types of labels generated
    """

    # get values of each percentile
    list_percentile_values = [df[var_segment].quantile(x).round(2) for x in list_percentile]
    
    # generate a list of string with the start value and end value of each interval
    list_percentile_start_end = [] 
    for index in range(len(list_percentile_values)-1): 
        start_value = list_percentile_values[index]
        end_value = list_percentile_values[index+1]
        string_start_end = f'{start_value}-{end_value}'
        list_percentile_start_end.append(string_start_end)
    
    # output final v0 - base
    #list_labels_percentile_base
    
    # output final v1 - only values start end
    list_labels_values_range = []
    for index in range(len(list_labels_percentile_base)):
        string_output = f'({list_percentile_start_end[index]})'
        list_labels_values_range.append(string_output)
    
    # output final v2 - percentile and values start end
    list_labels_percentile_values_range = []
    for index in range(len(list_labels_percentile_base)):
        string_output = f'{list_labels_percentile_base[index]} - ({list_percentile_start_end[index]})'
        list_labels_percentile_values_range.append(string_output)
    
    return list_labels_percentile_base, list_labels_values_range, list_labels_percentile_values_range

def percentile_segmentation(df, var_segment, type_percentile):
    """
    Given a dataframe, generate a new column with a categorical values that divide the data in differents segments. 
    Segment the data by a certain variable with a percentile segmentation. the segmentation could be by quartiles, quintiles, deciles
    
    Args
        df (dataframe): dataframe input that will be modified
        var_segment (string): variable feature/target used to segment the data
        type_percentile(string): type of percentile segmentation
    
    Return
        df(dataframe): the input dataframe with a new column with the segment

    TODO: THE LABELS GERATED AND USED ARE ONLY ['q1 - (0.15-1.2)', 'q2 - (1.2-1.8)', 'q3 - (1.8-2.65)', 'q4 - (2.65-5.0)']
    ADD A ARGS TO SELECT THE KIND OF LABELS
    """

    # validate input - TODO: create a decent unit test
    choices_segmentation = ['quartile', 'quintile', 'decile']
    if type_percentile not in choices_segmentation:
        print('error in choices of segmentation')
        print(f'Possibles choices: {choices_segmentation}')
        return 0

    # quartile
    if type_percentile == 'quartile':
        quartile = [0, 0.25, 0.5, 0.75, 1]
        labels_quartile_base = ['q1', 'q2', 'q3', 'q4']
        _, _,  labels_quartile = generate_labels_percentile_segmentation(df, var_segment, quartile, labels_quartile_base)
        df[f'quartile_{var_segment}'] = pd.qcut(df[var_segment], q = quartile, labels = labels_quartile)
    
    # quintile
    if type_percentile == 'quintile':
        quintile = [0, 0.2, 0.4, 0.6, 0.8, 1]
        labels_quintile_base = ['q1', 'q2', 'q3', 'q4', 'q5']
        _, _,  labels_quintile = generate_labels_percentile_segmentation(df, var_segment, quintile, labels_quintile_base)
        df[f'quintile_{var_segment}'] = pd.qcut(df[var_segment], q = quintile, labels = labels_quintile)


    # decile
    if type_percentile == 'decile':
        decile = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        labels_decile_base = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10']
        _, _,  labels_decile = generate_labels_percentile_segmentation(df, var_segment, decile, labels_decile_base)
        df[f'decile_{var_segment}'] = pd.qcut(df[var_segment], q = decile, labels = labels_decile)

    return df


##############################################  3.12 Categorical Analysis - freq each feature againts freq target ##############################################
def descriptive_statistics_target_for_each_feature(df, target):
    """
    Calculate descriptive statistics of target for each feature categorical (and for each category in each feature)

    Args:
        df (dataframe): input dataframe
        target (string): string to target that will be calcuated its statistics

    Return:
        df_statistics_target (dataframe): dataframe with the statistics of the target
        df_statistics_target_to_plotly (dataframe): dataframe with the statistics of the target adapted to show in a plotly graph
    """
    
    ### list_features
    list_features = list(set(df.columns.tolist()) - set([target]))
    
    ###### generate descriptive statistics of the target for each percentil of each feature
    df_statistics_target = pd.DataFrame()
    for feature in list_features:
        #print(feature)
        
        # calculate statistic descriptive of target for a categories of a feature
        aux_statistics_target = df.groupby(feature)[target].describe()
        
        # set multiindex (feature, percentile_feature)
        aux_statistics_target.index = pd.MultiIndex.from_product([
            [feature], 
            aux_statistics_target.index.tolist()
        ] )
        
        # join in a unique dataframe
        df_statistics_target = pd.concat([df_statistics_target, aux_statistics_target], axis = 0)
    
    
    ##### round to 3 decimals
    df_statistics_target.round(2)

    
    return df_statistics_target


##############################################  3.13 Categorical Analysis - crosstab freq target vs each feature ##############################################
def crosstab_freq_target_1_feature(df, feature, target):
    """
    Calculate a cross tab of frecuency of target (categorical) given one categorical feature.
    The output are 2 dataframes, the first is the output of pd.crosstab() and the second one is the previous output transformed to plot in plotly

    Args:
        df (dataframe): input dataframe with feature and target categorical variables
        feauture (string): name categorical variable to compare target
        target (string): name categorical target

    Return
        ct_freq_target (dataframe): cross tab of frecuency of target given according a categorical feature
        ct_freq_target_reset_index (dataframe): previous dataframe with transformations to plot in plotly barplot
    """

    ##### calculate cross tab
    # calculate cross table freq
    ct_freq_target = pd.crosstab(index = df[feature], columns = df[target])

    
    ##### transform into cross table accepted to plotly
    # reset index  to plot
    ct_freq_target_reset_index = ct_freq_target.reset_index()
    
    # convert table into a format to plotly express
    ct_freq_target_reset_index = pd.melt(ct_freq_target_reset_index, id_vars = feature, value_name='freq_target')

    return ct_freq_target, ct_freq_target_reset_index

def barplot_crosstab_freq_target_1_features(df, target, number_columns = 1):
    """
    Given a dataframe with columns features + target. Genereate a barplot of relations between each features and the freq of the target
    Detail: 
        Given a dataframe with features categorical, generate a crosstab of freq of target between feature and plot it in a barplot
        Calling a function to generate a cross table and then plot it with plotly
    
    Args
        df (dataframe): input dataframe with columns features and target
        target (string): target of the dataframe, column that will be delete to plot the relations between only features
        number_columns (integer): number of columns. because heatmap could be bigger, plot it into 1 columns by default

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    ################# generate a list of tuples of each pair of features to generate the cross table  #####################
    list_features = list(set(df.columns.tolist()) - set([target]))

    
    ####################### plot #################################
    
    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_features) % number_columns) != 0:
        number_rows = (len(list_features) // number_columns) + 1
    else:
        number_rows = (len(list_features) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, 
                        subplot_titles = tuple([str(tupla) for tupla in list_features]), ### title of each subplots
                        vertical_spacing = 0.02
                       )

    ########## for each tuple of features to plot:
    for index_feature, feature in enumerate(list_features):
        
        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        
        ## get cross table freq of target vs 1 categorical features - call the INDIVIDUAL FUNCTION TO GENERATE CROSS TABLE
        # the output are 2 dataframes, the first is the output of pd.crosstab() and the second one is the previous output transformed to plot in plotly
        _, ct_freq_target_plotly = crosstab_freq_target_1_feature(df = df, 
                                                         feature = feature, 
                                                         target = target)
        
        ## tranform cross table freq target vs one categorical feature into a barplot
        fig_barplot_aux = px.bar(ct_freq_target_plotly, 
                     x = feature, 
                     y='freq_target',
                     color = target,
                     barmode='group'
                    )
        
        # add barplot to fig global
        for index_plot in range(len(fig_barplot_aux.data)):
            fig.add_trace(fig_barplot_aux.data[index_plot],
                row = row,
                col = column
            )

    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo 400
        width = 1050 * number_columns,  # ancho  1850
        title_text =  f'freq of target:{target} vs each categorical feature individual',
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig


##############################################  3.1 statistics ##############################################
##############################################  3.1 statistics ##############################################
##############################################  3.1 statistics ##############################################
##############################################  3.1 statistics ##############################################
##############################################  3.1 statistics ##############################################
##############################################  3.1 statistics ##############################################