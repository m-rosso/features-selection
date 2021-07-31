####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np

import re
# pip install unidecode
from unidecode import unidecode

# pip install plotly==4.6.0
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# pip install cufflinks==0.17.3
import cufflinks as cf
cf.go_offline()

####################################################################################################################################
####################################################################################################################################
#############################################################FUNCTIONS##############################################################

####################################################################################################################################
# Function that splits data into train and test set:

def train_test_split(dataframe, test_ratio=0.5, shuffle=False, seed=None):
    """
    Function that splits data into train and test set.
    
    :param dataframe: complete set of data.
    :type dataframe: dataframe.
    :param seed: seed for shuffle.
    :type seed: integer.
    :param test_ratio: proportion of data to be allocated into test set.
    :type test_ratio: float.
    :param shuffle: indicates whether to shuffle data previously to the split.
    :type shuffle: boolean.
    
    :return: training and test dataframes.
    :rtype: tuple.
    """
    df = dataframe.copy()
    df.reset_index(drop=True, inplace=True)
    
    if shuffle:
        df = df.sample(len(df), random_state=seed)
    
    # Indexes for training and test data:
    test_indexes = [True if i > int(len(df)*(1 - 0.25)) else False for i in range(len(df))]
    train_indexes = [True if i==False else False for i in test_indexes]
    
    # Train-test split:
    df_train = df.iloc[train_indexes, :]
    df_test = df.iloc[test_indexes, :]
    
    return (df_train, df_test)

####################################################################################################################################
# Function that produces a dataframe with frequency of features by class and returns lists with features names by class:

def classify_variables(dataframe, vars_to_drop=[], drop_excessive_miss=True, excessive_miss=0.95,
                       drop_no_var=True, validation_data=None, test_data=None):
    """
    Function that produces a dataframe with frequency of features by class and returns lists with features names by class.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :param vars_to_drop: list of support columns.
    :type vars_to_drop: list.

    :param drop_excessive_miss: flag indicating whether columns with excessive missings should be dropped out.
    :type drop_excessive_miss: boolean.

    :param excessive_miss: share of missings above which columns are dropped from the dataframes.
    :type excessive_miss: float.

    :param drop_no_var: flag indicating whether columns with no variance should be dropped out.
    :type drop_no_var: boolean.

    :param validation_data: additional data.
    :type validation_data: dataframe.

    :param test_data: additional data.
    :type test_data: dataframe.

    :return: dataframe and lists with features by class.
    :rtype: dictionary.
    """
    print(f'Initial number of features: {dataframe.drop(vars_to_drop, axis=1).shape[1]}.')

    if drop_excessive_miss:
        # Dropping features with more than 95% of missings in the training data:
        excessive_miss_train = [c for c in dataframe.drop(vars_to_drop, axis=1) if
                                sum(dataframe[c].isnull())/len(dataframe) > excessive_miss]

        if len(excessive_miss_train) > 0:
            dataframe.drop(excessive_miss_train, axis=1, inplace=True)

            if validation_data is not None:
                validation_data.drop(excessive_miss_train, axis=1, inplace=True)
                
            if test_data is not None:
                test_data.drop(excessive_miss_train, axis=1, inplace=True)

        print(f'{len(excessive_miss_train)} features were dropped for excessive number of missings!')
        
    # Data type of each variable:
    type_vars = dict(zip(dataframe.drop(vars_to_drop, axis=1).dtypes.index,
                         dataframe.drop(vars_to_drop, axis=1).dtypes.values))
    
    # Classifying features:
    cat_vars = []
    binary_vars = []
    cont_vars = []

    # Loop over variables:
    for v in type_vars.keys():
        # Categorical features:
        if type_vars[v] == object:
            cat_vars.append(v)

        # Numerical features:
        else:
            # Binary variables:
            if (dataframe[v].nunique() == 2) & (sorted(dataframe[v].unique()) == [0, 1]):
                binary_vars.append(v)

            # Continuous variables:
            else:
                cont_vars.append(v)
        
    if drop_no_var:
        # Dropping features with no variance in the training data:
        no_variance = [c for c in dataframe.drop(vars_to_drop, axis=1).drop(cat_vars,
                                                                         axis=1) if np.nanvar(dataframe[c])==0]

        if len(no_variance) > 0:
            dataframe.drop(no_variance, axis=1, inplace=True)
            if validation_data is not None:
                validation_data.drop(no_variance, axis=1, inplace=True)
                
            if test_data is not None:
                test_data.drop(no_variance, axis=1, inplace=True)

        print(f'{len(no_variance)} features were dropped for having no variance!')
        
    print(f'{dataframe.drop(vars_to_drop, axis=1).shape[1]} remaining features.')
    print('\n')
    
    # Dataframe presenting the frequency of features by class:
    feats_assess = pd.DataFrame(data={
        'class': ['cat_vars', 'binary_vars', 'cont_vars', 'vars_to_drop'],
        'frequency': [len(cat_vars), len(binary_vars), len(cont_vars), len(vars_to_drop)]
    })
    feats_assess.sort_values('frequency', ascending=False, inplace=True)
    
    # Dictionary with outputs from the function:
    feats_assess_dict = {
        'feats_assess': feats_assess,
        'cat_vars': cat_vars,
        'binary_vars': binary_vars,
        'cont_vars': cont_vars
    }
    
    if drop_excessive_miss:
        feats_assess_dict['excessive_miss_train'] = excessive_miss_train

    if drop_no_var:
        feats_assess_dict['no_variance'] = no_variance
    
    return feats_assess_dict

####################################################################################################################################
# Function that returns the amount of time for running a block of code:

def running_time(start_time, end_time, print_time=True):
    """
    Function that returns the amount of time for running a block of code.
    
    :param start_time: time point when the code was initialized.
    :type start_time: datetime object obtained by executing "datetime.now()".

    :param end_time: time point when the code stopped its execution.
    :type end_time: datetime object obtained by executing "datetime.now()".

    :param print_unit: unit of time for presenting the runnning time.
    :type print_unit: string.
    
    :return: prints start, end time and running times, besides of returning running time in seconds.
    :rtype: integer.
    """
    if print_time:
        print('------------------------------------')
        print('\033[1mRunning time:\033[0m ' + str(round(((end_time - start_time).total_seconds())/60, 2)) +
              ' minutes.')
        print('Start time: ' + start_time.strftime('%Y-%m-%d') + ', ' + start_time.strftime('%H:%M:%S'))
        print('End time: ' + end_time.strftime('%Y-%m-%d') + ', ' + end_time.strftime('%H:%M:%S'))
        print('------------------------------------')
    
    return (end_time - start_time).total_seconds()

####################################################################################################################################
# Function that calculates cross-entropy given true labels and predictions:

def cross_entropy_loss(y_true, p):
    prediction = np.clip(p, 1e-14, 1. - 1e-14)
    return -np.sum(y_true*np.log(prediction) + (1-y_true)*np.log(1-prediction))/len(y_true)

####################################################################################################################################
# Function for cleaning texts:

def text_clean(text, lower=True):
    if pd.isnull(text):
        return text
    
    else:
        text = str(text)

        # Removing accent:
        text_cleaned = unidecode(text)
        # try:
        #     text_cleaned = unidecode(text)
        # except AttributeError as error:
        #     print(f'Error: {error}.')

        # Removing extra spaces:
        text_cleaned = re.sub(' +', ' ', text_cleaned)
        
        # Removing spaces before and after text:
        text_cleaned = str.strip(text_cleaned)
        
        # Replacing spaces:
        text_cleaned = text_cleaned.replace(' ', '_')
        
        # Deleting signs:
        for m in '.,;+-!@#$%¨&*()[]{}\\/|':
            if m in text_cleaned:
                text_cleaned = text_cleaned.replace(m, '')

        # Setting text to lower case:
        if lower:
            text_cleaned = text_cleaned.lower()

        return text_cleaned

####################################################################################################################################
# Function that produces an assessment of the occurrence of missing values:

def assessing_missings(dataframe):
    """
    Function that produces an assessment of the occurrence of missing values.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :return: dataframe with frequency and share of missings by feature.
    :rtype: dataframe.
    """
    # Dataframe with the number of missings by feature:
    missings_dict = dataframe.isnull().sum().sort_values(ascending=False).to_dict()

    missings_df = pd.DataFrame(data={
        'feature': list(missings_dict.keys()),
        'missings': list(missings_dict.values()),
        'share': [m/len(dataframe) for m in list(missings_dict.values())]
    })

    print('\033[1mNumber of features with missings:\033[0m {}'.format(sum(missings_df.missings > 0)) +
          ' out of {} features'.format(len(missings_df)) +
          ' ({}%).'.format(round((sum(missings_df.missings > 0)/len(missings_df))*100, 2)))
    print('\033[1mAverage number of missings:\033[0m {}'.format(int(missings_df.missings.mean())) +
          ' out of {} observations'.format(len(dataframe)) +
          ' ({}%).'.format(round((int(missings_df.missings.mean())/len(dataframe))*100,2)))
    
    return missings_df

####################################################################################################################################
# Function that assess the number of missings in a dataframe:

def missings_detection(dataframe, name='df', var=None):
    """"
    Function that assess the number of missings in a dataframe

    :param dataframe: dataframe for which missings should be detected.
    :type dataframe: dataframe.
    
    :param name: name of dataframe.
    :type name: string.
    
    :param var: name of variable whose missings should be detected (optional).
    :type var: string.

    :return: prints the number of missings when there is a positive amount of them.
    """

    if var:
        num_miss = dataframe[var].isnull().sum()
        if num_miss > 0:
            print(f'Problem - There are {num_miss} missings for "{var}" in dataframe {name}.')

    else:
        num_miss = dataframe.isnull().sum().sum()
        if num_miss > 0:
            print(f'Problem - Number of overall missings detected in dataframe {name}: {num_miss}.')

####################################################################################################################################
# Function that forces consistency between reference (training) and additional (validation, test) data:

def data_consistency(dataframe, *args, **kwargs):
    """
    Function that forces consistency between reference (training) and additional (validation, test) data:

    The keyword arguments are expected to be dataframes whose argument names indicate the nature of the passed data. For instance,
    'test_data=df_test' would be a dataframe with test instances.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :return: dataframes with consistent data.
    :rtype: dictionary.
    """
    consistent_data = {}
    
    for d in kwargs.keys():
        consistent_data[d] = kwargs[d].copy()
        
        # Columns absent in reference data:
        absent_train = [c for c in kwargs[d].columns if c not in dataframe.columns]
        
        # Columns absent in additional data:
        absent_test = [c for c in dataframe.columns if c not in kwargs[d].columns]
        
        # Creating absent columns:
        for c in absent_test:
            consistent_data[d][c] = 0
    
        # Preserving consistency between reference and additional data:
        consistent_data[d] = consistent_data[d][dataframe.columns]
        
        # Checking consistency:
        if sum([1 for r, a in zip(dataframe.columns, consistent_data[d].columns) if r != a]):
            print('Problem - Reference and additional datasets are inconsistent!')
        else:
            print(f'Training and {d.replace("_", " ")} are consistent with each other.')
    
    return consistent_data

####################################################################################################################################
# Function that plots different visualizations of outcomes:

def plot_outcomes(outcomes_lasso, outcomes_xgboost, plot):
    """
    Function that plots different visualizations of outcomes.
    
    :param outcomes_lasso: outcomes of Lasso estimation.
    :type outcomes_lasso: dataframe.
    :param outcomes_xgboost: outcomes of XGBoost estimation.
    :type outcomes_xgboost: dataframe.
    :param plot: name of the plot ("metric_by_approach", "ratio_by_approach", "metric_by_time",
    "metric_by_num_feats").
    :type plot: string.

    :return: a plotly graph ac
    """
    methods = ['lasso', 'xgboost']
    color_dict = {'lasso': '#c56d00', 'xgboost': '#01988d'}
    
    if plot=='metric_by_approach':
        # Create figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[])

        # Add bar plot:
        for m in range(len(methods)):
            outcomes = eval(f'outcomes_{methods[m]}').sort_values('test_r2', ascending=False)

            fig.add_trace(go.Bar(x=outcomes['features_selection'],
                                 y=outcomes['test_r2'],
                                 hovertemplate='test_r2 = %{y:.4f}<br>' + 'features_selection = %{x}<br>' + '%{text}<br>',
                                 text=[f'running_time = {x:.2f}<br>num_selected_features = {y}' for x, y in
                                       zip(outcomes['running_time'],
                                           outcomes['num_selected_features'])],
                                 name=methods[m],
                                 marker_color=color_dict[methods[m]]),
                          row=1, col=m+1)

        # Changing overall layout:
        fig.update_layout(title_text='Performance metric by features selection approach',
                          width=900, height=500,
                          barmode='group')

        # Set labels:
        fig.update_xaxes(title_text='Features selection', showticklabels=False)

    elif plot=='ratio_by_approach':
        # Create figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[])

        # Add bar plot:
        for m in range(len(methods)):
            outcomes = eval(f'outcomes_{methods[m]}').sort_values('ratio_r2_time', ascending=False)

            fig.add_trace(go.Bar(x=outcomes['features_selection'],
                                 y=outcomes['ratio_r2_time'],
                                 hovertemplate='%{text}<br>',
                                 text=[f'test_r2 = {w:.2f}<br>features_selection = {x}<br>running_time = {y:.2f}<br>num_selected_features = {z}'
                                       for w, x, y, z in zip(outcomes['test_r2'],
                                                             outcomes['features_selection'],
                                                             outcomes['running_time'],
                                                             outcomes['num_selected_features'])],
                                 name=methods[m],
                                 marker_color=color_dict[methods[m]]),
                          row=1, col=m+1)

        # Changing overall layout:
        fig.update_layout(title_text='Ratio between performance metric and running time by features selection approach',
                          width=900, height=500,
                          barmode='group')

        # Set labels:
        fig.update_xaxes(title_text='Features selection', showticklabels=False)

    elif plot=='metric_by_time':
        # Create figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[])

        # Add bar plot:
        for m in range(len(methods)):
            outcomes = eval(f'outcomes_{methods[m]}')

            fig.add_trace(go.Scatter(x=outcomes['running_time'],
                                     y=outcomes['test_r2'],
                                     hovertemplate='test_r2 = %{y:.4f}<br>' + 'running_time = %{x:.2f}<br>' +
                                                   '%{text}<br>',
                                     text=[f'num_selected_features = {x}<br>features_selection = {y}' for x, y in
                                           zip(outcomes['num_selected_features'],
                                               outcomes['features_selection'])],
                                     name=methods[m],
                                     mode='markers', marker_color=color_dict[methods[m]]),
                          row=1, col=m+1)

        # Changing overall layout:
        fig.update_layout(title_text='Performance metric against running time',
                          width=900, height=500,
                          barmode='group')

        # Set labels:
        fig.update_xaxes(title_text='Running time (seconds)')

    elif plot=='metric_by_num_feats':
        # Create figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[])

        # Add bar plot:
        for m in range(len(methods)):
            outcomes = eval(f'outcomes_{methods[m]}')

            fig.add_trace(go.Scatter(x=outcomes['num_selected_features'],
                                     y=outcomes['test_r2'],
                                     hovertemplate='test_r2 = %{y:.4f}<br>' + 'num_selected_features = %{x}<br>' +
                                                   '%{text}<br>',
                                     text=[f'running_time = {x}<br>features_selection = {y}' for x, y in
                                           zip(outcomes['running_time'],
                                               outcomes['features_selection'])],
                                     name=methods[m],
                                     mode='markers', marker_color=color_dict[methods[m]]),
                          row=1, col=m+1)

        # Changing overall layout:
        fig.update_layout(title_text='Performance metric against the number of selected features',
                          width=900, height=500,
                          barmode='group')

        # Set labels:
        fig.update_xaxes(title_text='Number of selected features')

    else:
        return
    
    # Set labels:
    fig.update_yaxes(title_text='Test R2', row=1, col=1)
    fig.update_yaxes(title_text='', row=1, col=2)
        
    # Printing the plot:
    fig.show()
