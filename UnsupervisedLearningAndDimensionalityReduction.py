import datetime
import json
import os
from os import path
import random
import sys
import time

import itertools
import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy import linalg
from scipy.io import arff
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
import sklearn.cluster as cluster
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn import mixture
from sklearn import random_projection
from scipy import stats

import Utils

###############################################################
################ Dataset Initialization Methods ###############
###############################################################
def initializeDataset(datasetDict, randomState):
    if datasetDict['name'] == 'starcraft':
        initializeStarcraftDataset(datasetDict, randomState)
    elif datasetDict['name'] == 'seismic bumps':
        initializeSeismicBumpsDataset(datasetDict, randomState)


def initializeStarcraftDataset(datasetDict, randomState):
    # load first arff file
    data_1, meta_1 = arff.loadarff('datasets/starcraft/scmPvT_Protoss_Mid.arff')
    df_1 = pd.DataFrame(data_1)
    df_1['opponent'] = pd.Series(np.full(df_1.shape[0], 0))

    # load second arff file
    data_2, meta_2 = arff.loadarff('datasets/starcraft/scmPvZ_Protoss_Mid.arff')
    df_2 = pd.DataFrame(data_2)
    df_2['opponent'] = pd.Series(np.full(df_2.shape[0], 1))

    # concatenate the two dataframes into one
    df = pd.concat([df_1, df_2])

    # pull out the features and target variable
    X = df.loc[:, df.columns != 'midBuild']
    datasetDict['featureNames'] = list(X.columns)
    #print(X.head())
    Y = df['midBuild'].to_frame()
    #print(Y.head())
    #print(Y['midBuild'].value_counts())

    datasetDict['cv'] = StratifiedKFold(n_splits=4, random_state=randomState, shuffle=True)
    datasetDict['shuffle'] = True
    datasetDict['trainPercentage'] = .80

    # 680 was used when we were just using one data file
    # dataset_dict[train_sizes_key] = [int(x) for x in np.arange(8, 680, 10)]
    datasetDict['trainSizes'] = [int(x) for x in np.arange(8, 1295, 25)]

    datasetDict['outputClassNames'] = ['Carrier', 'FastDT', 'FastExpand', 'FastLegs', 'FastObs', 'ReaverDrop',
                                            'Unknown']

    # do the one hot encoding using pandas instead of sklearn
    # X_enc = pd.get_dummies(X)
    # print(X_enc.head())
    X_enc = X

    # standardize because one of the features goes from 0-1 while all the others are on the same scale
    # standardizing doesn't hurt
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
    X_std = scaler.fit_transform(X_enc)

    # Some machine learning classifiers in Scikit-learn prefer that the class labels in the target variable are encoded
    # with numbers. Since we only have two classes, we can use LabelEncoder.
    le = preprocessing.LabelEncoder()
    Y_enc = le.fit_transform(Y.values.ravel())

    X = X_std
    Y = Y_enc

    datasetDict['X'] = X
    datasetDict['Y'] = Y


def initializeSeismicBumpsDataset(datasetDict, randomState):
    # code from http://nmouatta.blogspot.com/2016/09/imbalanced-class-classification-with.html

    # load arff file
    data, meta = arff.loadarff('datasets/seismic-bumps/seismic-bumps.arff')
    df = pd.DataFrame(data)
    column_labels = ['seismic',
                'seismoacoustic',
                'shift',
                'genergy',
                'gpuls',
                'gdenergy',
                'gdpuls',
                'ghazard',
                'nbumps',
                'nbumps2',
                'nbumps3',
                'nbumps4',
                'nbumps5',
                'nbumps6',
                'nbumps7',
                'nbumps89',
                'energy',
                'maxenergy',
                'outcome']
    df.columns = column_labels
    #print(df.head())

    # pull out the features and target variable
    X = df.loc[:, df.columns != 'outcome']
    #print(X.head())
    Y = df['outcome'].to_frame()
    #print(Y.head())
    #print(Y['outcome'].value_counts())

    datasetDict['cv'] = StratifiedKFold(n_splits=4, random_state=randomState, shuffle=True)
    datasetDict['shuffle'] = True
    datasetDict['trainPercentage'] = .80

    # with a cv of 4
    datasetDict['trainSizes'] = [int(x) for x in np.arange(20, int(1549), 25)]

    datasetDict['outputClassNames'] = ['Non-Hazardous', 'Hazardous']

    datasetDict['scoring'] = 'f1'

    # do the one hot encoding using pandas instead of sklearn
    X_enc = pd.get_dummies(X)
    datasetDict['featureNames'] = list(X_enc.columns)

    #print(X_enc.head())

    # standardizing doesn't hurt
    #scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
    scaler = preprocessing.RobustScaler()
    X_std = scaler.fit_transform(X_enc)

    # Some machine learning classifiers in Scikit-learn prefer that the class labels in the target variable are encoded
    # with numbers. Since we only have two classes, we can use LabelEncoder.
    le = preprocessing.LabelEncoder()
    Y_enc = le.fit_transform(Y.values.ravel())

    X = X_std
    Y = Y_enc

    datasetDict['X'] = X
    datasetDict['Y'] = Y



###############################################################
####### Neural Network and Support Methods for it #############
###############################################################
# code taken from my assignment #1 and modified
def runANN(config, trainX, trainY, testX, testY, datasetName, datasetDict, datasetResultsDict, randomState):
    print("Running " + config['name'] + " on " + datasetName)
    algResultsDict = {}
    algName = config['name']
    datasetResultsDict[algName] = algResultsDict

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # load dataset/algorithm specific config information (if any)
    runLearningCurve = config['nnRunLearningCurve']
    runModelComplexityCurve = config['nnRunModelComplexityCurve']
    runFinalTest = config['nnRunFinalTest']
    runGridSearchCV = config['nnRunGridSearchCV']

    initialAlpha = config['nnInitialAlpha']
    maxIterations = config['nnMaxIterations']

    # load dataset specific configuration setting
    if 'scoring' in datasetDict.keys():
        scoring = datasetDict['scoring']
    else:
        scoring = None

    shuffle = datasetDict['shuffle']
    cv = datasetDict['cv']

    gs_best_nn = None

    ###########################################################
    # do the learning curve
    ###########################################################
    if runLearningCurve:
        learning_curve_results_dict = {}
        algResultsDict['learningCurve'] = learning_curve_results_dict
        learning_curve_train_sizes = datasetDict['trainSizes']

        learning_curve_results_dict['learningCurveEstimatorSettings'] = {}

        mlp = MLPClassifier(alpha=initialAlpha, random_state=randomState, max_iter=maxIterations)

        if True or datasetName == "starcraft":
            first_param_value = mlp.get_params()['alpha']
            if first_param_value is not None:
                first_param_value = str(float(first_param_value))
            else:
                first_param_value = 'None'
            second_param_value = mlp.get_params()['learning_rate']
            if second_param_value is not None:
                second_param_value = second_param_value
            else:
                second_param_value = 'None'

            doLearningCurves(mlp, algName, trainX, trainY, cv,
                               learning_curve_train_sizes, shuffle,
                               scoring,
                               resultsDirectory, datasetName, learning_curve_results_dict,
                               ['alpha', 'learning_rate'], [first_param_value, second_param_value], "(Initial) ", True, randomState)
        else:
            print("unknown nn dataset")
            sys.exit()

        ###########################################################
        # now do the iteration 'training' curve
        ###########################################################
        if True or datasetName == "starcraft":
            training_size = 1000
            num_iterations = range(1, 2000, 20)
        else:
            print("unknown nn dataset")
            sys.exit()

        doIterationLearningCurves(mlp, algName, trainX, trainY, cv,
                           training_size, shuffle,
                           scoring,
                           resultsDirectory, datasetName, learning_curve_results_dict,
                           ['alpha', 'learning_rate'], [first_param_value, second_param_value], "(Initial) ", True, num_iterations, randomState)

    ###########################################################
    # now do the validation curve
    ###########################################################
    if runModelComplexityCurve:
        validation_curve_results_dict = {}
        algResultsDict['validationCurve'] = validation_curve_results_dict
        model_complexity_results_dict = {}
        algResultsDict['modelComplexity'] = model_complexity_results_dict

        if runGridSearchCV:
            gs_nn = MLPClassifier(alpha=initialAlpha, random_state=randomState, max_iter=maxIterations)

            num_nodes_in_layer_values = np.arange(1, 200, 4)
            num_hidden_layers_values = [1, 2]
            param_grid = [
                {'num_nodes_in_layer': num_nodes_in_layer_values, 'num_hidden_layers': num_hidden_layers_values}
            ]

            grid_param_1 = num_nodes_in_layer_values
            grid_param_2 = num_hidden_layers_values
            grid_param_1_name = 'alpha'
            grid_param_2_name = 'hidden_layer_sizes'

            gs_best_nn = doGridSearchCurves(gs_nn, algName, trainX, trainY, cv, param_grid, scoring,
                                  grid_param_1, grid_param_2, grid_param_1_name, grid_param_2_name, resultsDirectory, None, model_complexity_results_dict)

    ###########################################################
    # now do the full training of the final model and test on the test set
    ###########################################################
    if runFinalTest:
        final_test_results_dict = {}
        algResultsDict['finalTest'] = final_test_results_dict

        if gs_best_nn is None:
            finalAlpha = config['nnFinalAlpha']
            finalMaxIterations = config['nnFinalMaxIterations']
            finalNumHiddenLayers = config['nnFinalNumHiddenLayers']
            finalNumNodesPerLayer = config['nnFinalNumNodesPerLayer']

            #hidden_layer_sizes_values = []
            if finalNumHiddenLayers == 1:
                hidden_layer_sizes_values = (finalNumNodesPerLayer)
            elif finalNumHiddenLayers == 2:
                hidden_layer_sizes_values = (finalNumNodesPerLayer, finalNumNodesPerLayer)
            else:
                print("Too many hidden layers!!")
                sys.exit()

            gs_best_nn = MLPClassifier(alpha=finalAlpha, random_state=randomState, max_iter=finalMaxIterations, hidden_layer_sizes=hidden_layer_sizes_values)
            gs_best_nn.fit(trainX, trainY)

        start_time = time.time()
        Y_test_pred = gs_best_nn.predict(testX)
        end_time = time.time()
        final_test_results_dict['finalPredictTime'] = end_time - start_time

        final_test_results_dict['finalTestParams'] = gs_best_nn.get_params()

        #########################################################
        # do the final learning curve
        #########################################################
        if runLearningCurve:
            first_param_value = gs_best_nn.get_params()['alpha']
            if first_param_value is not None:
                first_param_value = str(float(first_param_value))
            else:
                first_param_value = 'None'
            second_param_value = gs_best_nn.get_params()['learning_rate']
            if second_param_value is not None:
                second_param_value = second_param_value
            else:
                second_param_value = 'None'

            doLearningCurves(gs_best_nn, algName, trainX, trainY, cv,
                               learning_curve_train_sizes, shuffle,
                               scoring,
                               resultsDirectory, datasetName, learning_curve_results_dict,
                               ['alpha', 'learning_rate'], [first_param_value, second_param_value], "(Final) ", False, randomState)

        ###########################################################
        # now do the final iteration 'training' curve
        ###########################################################
        if runLearningCurve:
            training_size = 1000
            num_iterations = range(1, 2000, 20)

            doIterationLearningCurves(gs_best_nn, algName, trainX, trainY, cv,
                               training_size, shuffle,
                               scoring,
                               resultsDirectory, datasetName, learning_curve_results_dict,
                               ['alpha', 'learning_rate'], [first_param_value, second_param_value], "(Final) ", True, num_iterations, randomState)


            # reset this back to what it was before doing the iteration learning curves
            gs_best_nn.set_params(max_iter=maxIterations)

        ##########################################################
        # output the confusion matrix and other final errors
        ##########################################################

        test_confusion_matrix = confusion_matrix(testY, Y_test_pred)

        classes = datasetDict['outputClassNames']

        # Plot non-normalized confusion matrix
        plt.figure()
        Utils.plot_confusion_matrix(test_confusion_matrix, classes=classes)

        plt.tight_layout()
        plt.savefig(path.join(resultsDirectory, algName + '_FinalTestConfusionMatrixNonNormalized.png'))
        plt.show()

        # Plot normalized confusion matrix
        plt.figure()
        Utils.plot_confusion_matrix(test_confusion_matrix, classes=classes, normalize=True)

        plt.tight_layout()
        plt.savefig(path.join(resultsDirectory, algName + '_FinalTestConfusionMatrixNormalized.png'))
        plt.show()

        if 'scoring' in datasetDict.keys() and datasetDict['scoring'] == 'f1':
            test_error = 1.0 - f1_score(testY, Y_test_pred)
        else:
            test_error = 1.0 - gs_best_nn.score(testX, testY)

        with open(path.join(resultsDirectory, algName + '_FinalError.txt'),
                  'w') as f:
            f.write(str(test_error))
        final_test_results_dict['finalTestError'] = test_error

    return


def doLearningCurves(estimator, algorithm_name, X_train, Y_train, cv,
                       learning_curve_train_sizes, shuffle,
                       scoring,
                       dataset_results_directory, dataset_name, learning_curve_results_dict,
                       variable_names, variable_values_as_strings, prefix_string,
                       is_initial, randomState):

    start_time = time.time()
    train_sizes_abs, learning_train_scores, learning_validation_scores = learning_curve(estimator, X_train, Y_train,
                                                                                        cv=cv,
                                                                                        # train_sizes=np.arange(5, 1003, 10),
                                                                                        train_sizes=learning_curve_train_sizes,
                                                                                        random_state=randomState,
                                                                                        shuffle=shuffle,
                                                                                        scoring=scoring)
    end_time = time.time()
    if is_initial:
        learning_curve_results_dict['initialLearningCurveTime'] = end_time - start_time
    else:
        learning_curve_results_dict['finalLearningCurveTime'] = end_time - start_time

    fig = plt.figure()
    ax = fig.add_subplot(111)

    mean_learning_train_scores = np.mean([1.0 - x for x in learning_train_scores[:]], axis=1)
    std_learning_train_scores = np.std([1.0 - x for x in learning_train_scores[:]], axis=1)
    # min_learning_train_scores = np.min([1.0 - x for x in learning_train_scores[:]], axis=1)
    # max_learning_train_scores = np.max([1.0 - x for x in learning_train_scores[:]], axis=1)
    mean_learning_validation_scores = np.mean([1.0 - x for x in learning_validation_scores[:]], axis=1)
    std_learning_validation_scores = np.std([1.0 - x for x in learning_validation_scores[:]], axis=1)
    # min_learning_validation_scores = np.min([1.0 - x for x in learning_validation_scores[:]], axis=1)
    # max_learning_validation_scores = np.max([1.0 - x for x in learning_validation_scores[:]], axis=1)

    learning_curve_results_dict['trainSizes'] = train_sizes_abs.tolist()
    #learning_curve_results_dict[mean_learning_train_scores_key] = mean_learning_train_scores.tolist()
    #learning_curve_results_dict[std_learning_train_scores_key] = std_learning_train_scores.tolist()
    #learning_curve_results_dict[mean_learning_validation_scores_key] = mean_learning_validation_scores.tolist()
    #learning_curve_results_dict[std_learning_validation_scores_key] = std_learning_validation_scores.tolist()

    ax.fill_between(train_sizes_abs, mean_learning_train_scores - std_learning_train_scores,
                    mean_learning_train_scores + std_learning_train_scores, alpha=0.1, color='r')
    ax.fill_between(train_sizes_abs, mean_learning_validation_scores - std_learning_validation_scores,
                    mean_learning_validation_scores + std_learning_validation_scores, alpha=0.1, color='grey')

    # ax.errorbar(train_sizes_abs, mean_learning_train_scores, yerr=[max_learning_train_scores-mean_learning_train_scores, mean_learning_train_scores-min_learning_train_scores], label='Their CV Train', color='orange', linestyle='dashed')
    ax.plot(train_sizes_abs, mean_learning_train_scores, '-o', label='Train', color='orange', ms=3) #, linestyle='dashed')
    # ax.errorbar(train_sizes_abs, mean_learning_validation_scores, yerr=[max_learning_validation_scores-mean_learning_validation_scores, mean_learning_validation_scores-min_learning_validation_scores], label='Their CV Validation', color='black')
    ax.plot(train_sizes_abs, mean_learning_validation_scores, '-o', label='CV', color='black', ms=3)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('# Training Samples')
    ax.set_ylabel('Error')

    ax.minorticks_on()
    ax.grid(b=True, which='major', color='b', linestyle='-', alpha=0.2)
    ax.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)

    leg = ax.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.6)
    title_string = ""
    title_string += prefix_string + "Dataset: " + dataset_name + ", " + "Algorithm: " + algorithm_name + "\n"
    for variable_index, _ in enumerate(variable_names):
        title_string += variable_names[variable_index] + ": " + variable_values_as_strings[variable_index] + "\n"

    with open(path.join(dataset_results_directory, algorithm_name + '_' + prefix_string +'_LearningCurve.txt'), 'w') as f:
        f.write(json.dumps(title_string))

    plt.title(prefix_string + ' Learning Curve')
    plt.tight_layout()
    plt.savefig(path.join(dataset_results_directory, algorithm_name + '_' + prefix_string +'_LearningCurve.png'))
    plt.show()


# taken and modified from https://matplotlib.org/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py
def doIterationLearningCurves(estimator, algorithm_name, X_train, Y_train, cv,
                       learning_curve_train_size, shuffle,
                       scoring,
                       dataset_results_directory, dataset_name, learning_curve_results_dict,
                       variable_names, variable_values_as_strings, prefix_string,
                       is_initial, iterations, randomState):

    mean_learning_train_scores = []
    mean_learning_validation_scores = []

    start_time = time.time()
    for iteration in iterations:
        estimator.set_params(max_iter=iteration)
        train_sizes_abs, learning_train_scores, learning_validation_scores = learning_curve(estimator, X_train, Y_train,
                                                                                            cv=cv,
                                                                                            # train_sizes=np.arange(5, 1003, 10),
                                                                                            train_sizes=[learning_curve_train_size],
                                                                                            random_state=randomState,
                                                                                            shuffle=shuffle,
                                                                                            scoring=scoring)

        mean_learning_train_scores.extend(np.mean([1.0 - x for x in learning_train_scores[:]], axis=1))
        std_learning_train_scores = np.std([1.0 - x for x in learning_train_scores[:]], axis=1)
        mean_learning_validation_scores.extend(np.mean([1.0 - x for x in learning_validation_scores[:]], axis=1))
        std_learning_validation_scores = np.std([1.0 - x for x in learning_validation_scores[:]], axis=1)
        #print("Iteration: ", iteration, " mean error: ", mean_learning_validation_scores)

    end_time = time.time()
    if is_initial:
        learning_curve_results_dict['initialTrainingCurveTime'] = end_time - start_time
    else:
        learning_curve_results_dict['finalTrainingCurveTime'] = end_time - start_time

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.fill_between(iterations, mean_learning_train_scores - std_learning_train_scores,
                    mean_learning_train_scores + std_learning_train_scores, alpha=0.1, color='r')
    ax.fill_between(iterations, mean_learning_validation_scores - std_learning_validation_scores,
                    mean_learning_validation_scores + std_learning_validation_scores, alpha=0.1, color='grey')

    ax.plot(iterations, mean_learning_train_scores, '-o', label='Train', color='orange', ms=3) #, linestyle='dashed')
    ax.plot(iterations, mean_learning_validation_scores, '-o', label='CV', color='black', ms=3)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('# Iterations')
    ax.set_ylabel('Error')

    ax.minorticks_on()
    ax.grid(b=True, which='major', color='b', linestyle='-', alpha=0.2)
    ax.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)

    leg = ax.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.6)
    title_string = ""
    title_string += prefix_string + "Dataset: " + dataset_name + ", " + "Algorithm: " + algorithm_name + "\n"
    for variable_index, _ in enumerate(variable_names):
        title_string += variable_names[variable_index] + ": " + variable_values_as_strings[variable_index] + "\n"

    with open(path.join(dataset_results_directory, algorithm_name + '_' + prefix_string +'_TrainingCurve.txt'), 'w') as f:
        f.write(json.dumps(title_string))

    plt.title(prefix_string + ' Training Curve')
    plt.tight_layout()
    plt.savefig(path.join(dataset_results_directory, algorithm_name + '_' + prefix_string +'_TrainingCurve.png'))
    plt.show()


def doGridSearchCurves(estimator, algorithm_name, X_train, Y_train,
                         cv, param_grid, scoring,
                         grid_param_1, grid_param_2, grid_param_1_name, grid_param_2_name,
                         dataset_results_directory, x_axis_log_base, model_complexity_results_dict):

    # some of this is from http://scikit-learn.org/stable/modules/grid_search.html
    # we have to do something special with nn because my param grid has # hidden layers and # nodes
    # doesn't fit neatly into GridSearchCV's version of param_grid
    num_nodes_in_layer_values = np.arange(1, 200, 4)
    num_hidden_layers_values = [1, 2]

    hidden_layer_sizes_values = []
    for num_hidden_layers_value in num_hidden_layers_values:
        for num_nodes_in_layer_value in num_nodes_in_layer_values:
            if num_hidden_layers_value == 1:
                hidden_layer_sizes_values.append((num_nodes_in_layer_value,))
            else:
                hidden_layer_sizes_values.append((num_nodes_in_layer_value, num_nodes_in_layer_value))

    nn_param_grid = [
        {'hidden_layer_sizes': hidden_layer_sizes_values, 'alpha': [0.0001]}
    ]

    start_time = time.time()
    estimator = GridSearchCV(estimator, nn_param_grid, cv=cv, scoring=scoring, return_train_score=True)
    end_time = time.time()
    model_complexity_results_dict['modelComplexityTime'] = end_time - start_time

    # now fit the optimum model to the full training data
    start_time = time.time()
    estimator.fit(X_train, Y_train)
    end_time = time.time()
    model_complexity_results_dict['finalFitTime'] = end_time - start_time

    #print(estimator.cv_results_.keys())

    # Calling Method
    plotGridSearch(estimator.cv_results_, grid_param_1, grid_param_2, grid_param_1_name, grid_param_2_name,
                     algorithm_name, dataset_results_directory, x_axis_log_base)

    return estimator.best_estimator_ # this can be used to run the best estimator for final testing


# taken and modified from https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
def plotGridSearch(cv_results, grid_param_1, grid_param_2, grid_param_1_name, grid_param_2_name,
                     algorithm_name, dataset_results_directory, x_axis_log_base=None):
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    # same values as below

    # this seemed to be correct
    num_nodes_in_layer_values = np.arange(1, 200, 4)
    num_hidden_layers_values = [1, 2]

    # Get Test Scores Mean and std for each grid search
    validation_errors_mean = [1.0 - x for x in cv_results['mean_test_score']]
    validation_errors_mean = np.array(validation_errors_mean).reshape(len(num_hidden_layers_values), len(num_nodes_in_layer_values))

    training_errors_mean = [1.0 - x for x in cv_results['mean_train_score']]
    training_errors_mean = np.array(training_errors_mean).reshape(len(num_hidden_layers_values), len(num_nodes_in_layer_values))

    validation_scores_sd = cv_results['std_test_score']
    validation_scores_sd = np.array(validation_scores_sd).reshape(len(num_hidden_layers_values), len(num_nodes_in_layer_values))

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    for idx, val in enumerate(num_hidden_layers_values):
        ax.plot(num_nodes_in_layer_values, validation_errors_mean[idx,:], '-o', ms=3, label= 'CV '+  '# hidden layers' + '=' + str(val))
        ax.plot(num_nodes_in_layer_values, training_errors_mean[idx, :], '-o', ms=3, label='Train '+ '# hidden layers' + '=' + str(val))

    ax.set_title("Grid Search Validation Errors") #, fontsize=20, fontweight='bold')
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("# Nodes Per Layer")  # , fontsize=16)
    ax.set_ylabel('Train Error/CV Average Error') #, fontsize=16)

    ax.minorticks_on()
    ax.grid(b=True, which='major', color='b', linestyle='-', alpha=0.2)
    ax.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)

    leg = ax.legend(loc="best", fancybox=True) #, fontsize=15)
    leg.get_frame().set_alpha(0.6)

    if (x_axis_log_base is not None):
        ax.set_xscale("log", basex=x_axis_log_base)

    ax.grid('on')

    plt.tight_layout()
    plt.savefig(path.join(dataset_results_directory, algorithm_name + '_GridSearchValidation.png'))
    plt.show()


###############################################################
########### Dimensionality Reduction Methods ##################
###############################################################
# code taken and modified from https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/
# and https://blog.datadive.net/selecting-good-features-part-iii-random-forests/
def runRandomForestRegressor(config, X, Y, datasetName, datasetResultsDict, datasetDict, randomState):
    print("Running " + config['name'] + " on " + datasetName)
    algResultsDict = {}
    algName = config['name']
    datasetResultsDict[algName] = algResultsDict

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # fit random forest model
    model = RandomForestRegressor(n_estimators=500, random_state=randomState)
    model.fit(X, Y)
    # show importance scores
    # print(model.feature_importances_)
    # plot importance scores
    names = datasetDict['featureNames']
    ticks = [i for i in range(len(names))]
    pyplot.bar(ticks, model.feature_importances_)
    pyplot.xticks(ticks, names, rotation='vertical')
    # Pad margins so that markers don't get clipped by the axes
    #plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.40)

    if config['generateGraphs']:
        pyplot.savefig(path.join(resultsDirectory, config['name'] + 'FeatureImportance.png'))
        pyplot.show()

    print("Features sorted by their score:")
    featuresSortedByScoreDescending = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names),
           reverse=True)
    algResultsDict['featuresSortedByScoreDescending'] = featuresSortedByScoreDescending
    print(featuresSortedByScoreDescending)

    if 'rfMinFeatureValueToInclude' in config:
        minFeatureValue = config['rfMinFeatureValueToInclude']
        indexesOfFeaturesToExclude = [index for index, f in enumerate(model.feature_importances_) if f < minFeatureValue]
        rmTransformedX = np.delete(X, indexesOfFeaturesToExclude, axis=1)
        return rmTransformedX
    else:
        return X

# code taken and modified from http://scikit-learn.org/stable/auto_examples/classification/plot_lda.html#sphx-glr-auto-examples-classification-plot-lda-py
# code also taken from https://www.safaribooksonline.com/library/view/python-machine-learning/9781787125933/ch05s02.html
def runLDA(config, trainX, trainY, testX, testY, datasetName, datasetResultsDict, datasetDict, randomState):
    datasetName = datasetDict['name']
    print("Running " + config['name'] + " on " + datasetName)
    algName = config['name']
    datasetResultsDict = datasetDict['results']

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # load dataset/algorithm specific config information (if any)
    numFeaturesMin = config['ldaNumFeaturesMin']
    numFeaturesMax = config['ldaNumFeaturesMax']
    numFeaturesRange = range(numFeaturesMin, numFeaturesMax + 1)
    numAverages = config['ldaNumAverages']  # how often to repeat classification

    acc_clf1, acc_clf2 = [], []
    for numFeatures in numFeaturesRange:
        score_clf1, score_clf2 = 0, 0
        for _ in range(numAverages):
            #clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', n_components=numFeatures).fit(trainX, trainY)
            #clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None, n_components=numFeatures).fit(trainX, trainY)
            clf1 = LinearDiscriminantAnalysis(solver='svd', n_components=numFeatures).fit(trainX, trainY)
            clf2 = LinearDiscriminantAnalysis(solver='svd', n_components=numFeatures).fit(trainX, trainY)

            score_clf1 += clf1.score(testX, testY)
            score_clf2 += clf2.score(testX, testY)

        acc_clf1.append(score_clf1 / numAverages)
        acc_clf2.append(score_clf2 / numAverages)

    #features_samples_ratio = np.array(numFeaturesRange) / (trainX.shape)[0]
    features_samples_ratio = np.array(numFeaturesRange) / 1

    lda = LinearDiscriminantAnalysis(solver='svd', n_components=trainX.shape[1]).fit(trainX, trainY)
    ldaTransformedX = lda.transform(trainX)

    if config['generateGraphs']:
        # explained variance ratio
        explainedVarianceRatioArray = lda.explained_variance_ratio_
        cumulativeExplainedVarianceRatioArray = np.cumsum(explainedVarianceRatioArray)
        plt.plot(cumulativeExplainedVarianceRatioArray)
        plt.title("Cumulative Explained Variance for " + datasetName)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.grid()

        if config['generateGraphs']:
            plt.savefig(path.join(resultsDirectory, config['name'] + 'CumExplainedVariance.png'))
            plt.show()
        plt.close('all')

        # show scores (doesn't seem to work right)
        plt.plot(features_samples_ratio, acc_clf1, linewidth=2,
                 label="Linear Discriminant Analysis with shrinkage", color='navy')
        plt.plot(features_samples_ratio, acc_clf2, linewidth=2,
                 label="Linear Discriminant Analysis", color='gold')

        plt.xlabel('n_features')
        plt.ylabel('Classification accuracy')

        plt.legend(loc=1, prop={'size': 12})
        plt.suptitle('Linear Discriminant Analysis vs. \
        shrinkage Linear Discriminant Analysis (1 discriminative feature)')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_ClassificationAccuracy.png'))
        plt.show()

    return ldaTransformedX


def runRandomizedProjections(config, X, datasetDict, randomState):
    datasetName = datasetDict['name']
    print("Running " + config['name'] + " on " + datasetName)
    algName = config['name']
    datasetResultsDict = datasetDict['results']

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # load dataset/algorithm specific config information (if any)
    numComponentsMin = config['rpNumComponentsMin']
    numComponentsMax = config['rpNumComponentsMax']
    if 'rpBestRandomState' in config:
        bestRandomState = config['rpBestRandomState']
    else:
        bestRandomState = None
    numComponentsRange = range(numComponentsMin, numComponentsMax + 1)
    numTimesToRun = config['rpNumTimesToRun']

    bestXTransformed = None
    reconstructionErrorArrays = []
    for r in range(numTimesToRun):
        reconstructionErrorArrays.append([])
    for numComponents in numComponentsRange:
        for r in range(numTimesToRun):
            randomProjection = random_projection.GaussianRandomProjection(random_state=randomState+r, n_components=numComponents)
            XTransformed = randomProjection.fit_transform(X)
            if bestRandomState is not None and randomState + r == bestRandomState:
                bestXTransformed = XTransformed

            # code borrowed from Piazza post
            randMat = randomProjection.components_
            XProj = XTransformed.dot(randMat)
            XDiff = X - XProj
            XDiffSquared = XDiff * XDiff
            reconstructionError = np.sum(XDiffSquared)
            reconstructionErrorArrays[r].append(reconstructionError)

    if config['generateGraphs']:
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
        # plot reconstruction error
        for r in range (numTimesToRun):
            plt.plot(numComponentsRange, reconstructionErrorArrays[r], color=colors[r], label="RandomState="+str(randomState+r))
        plt.title("Reconstruction Error Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Number of Components')
        plt.ylabel('Reconstruction Error')
        plt.legend(loc=1, prop={'size': 12})
        plt.savefig(path.join(resultsDirectory, config['name'] + '_ReconstructionError.png'))
        plt.show()

    return  bestXTransformed


def oldrunRandomizedProjections(config, X, datasetDict, randomState):
    datasetName = datasetDict['name']
    print("Running " + config['name'] + " on " + datasetName)
    algName = config['name']
    datasetResultsDict = datasetDict['results']

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # load dataset/algorithm specific config information (if any)
    numComponentsMin = config['rpNumComponentsMin']
    numComponentsMax = config['rpNumComponentsMax']
    numComponentsRange = range(numComponentsMin, numComponentsMax + 1)
    numTimesToRun = config['rpNumTimesToRun']

    for numComponents in numComponentsRange:
        randomProjection = random_projection.GaussianRandomProjection(random_state=randomState, n_components=numComponents)
        X_transformed = randomProjection.fit_transform(X)
        #reconstructionError = np.matmul(randomProjection.components_, X_transformed)
        #reconstructionError = np.matmul(randomProjection.components_, X_transformed)
        #print(str(numComponents) + " Components Reconstruction error: ", str(reconstructionError))

    # this was used for eps based execution, but we won't use that anymore
    #epsScale = 100.0
    #scaledEPSMin = config['rpCcaledEPSMin']
    #scaledEPSMax = config['rpCcaledEPSMax']
    #scaledEPSStep = config['rpScaledEPSStep']
    #scaledEPSRange = range(scaledEPSMin, scaledEPSMax + scaledEPSStep, scaledEPSStep)
    #
    #for scaledEPS in scaledEPSRange:
        #randomProjection = random_projection.GaussianRandomProjection(random_state=randomState)
        #randomProjection = random_projection.SparseRandomProjection(random_state=randomState)
        #eps = None
        #eps = float(scaledEPS)/epsScale
        #randomProjection = random_projection.GaussianRandomProjection(random_state=randomState, eps=eps)
    #    randomProjection = random_projection.SparseRandomProjection(random_state=randomState, eps=eps)
    #    try:
    #        X_transformed = randomProjection.fit_transform(X)
    #        print(X_transformed.shape)
    #        reconstructionError = np.matmul(randomProjection.components_, X)
    #        print("Reconstruction error: ", str(reconstructionError))
    #    except Exception as e:
    #        print("RandomizedProject Exception " + str(e) + " for " + datasetName + " for eps: " + str(eps))
    return  X_transformed


def oldrunICA(config, X, datasetName, datasetResultsDict, datasetDict, randomState):
    print("Running " + config['name'] + " on " + datasetName)
    algResultsDict = {}
    algName = config['name']
    datasetResultsDict[algName] = algResultsDict

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # load dataset/algorithm specific config information (if any)
    maxIterations = config['icaMaxIterations']
    tolerance = config['icaTolerance']
    gFunction = config['icaGFunction']
    icaNumComponents = None
    if 'icaNumComponents' in config:
        icaNumComponents = config['icaNumComponents']
        ica = FastICA(n_components=icaNumComponents, random_state=randomState, max_iter=maxIterations, tol=tolerance, fun=gFunction, whiten=True)
    else:
        ica = FastICA(random_state=randomState, max_iter=maxIterations, tol=tolerance, fun=gFunction, whiten=True)
    icaTransformedX = ica.fit_transform(X)

    # kurtosis code copied from https://programtalk.com/python-examples/scipy.stats.kurtosis/
    kurtosisArray = stats.kurtosis(icaTransformedX)
    absKurtosisArray = np.absolute(kurtosisArray)
    # sorts by abs kurtosis value, descending
    sortedIndexArray = np.argsort(absKurtosisArray)[::-1]
    sortedAbsKurtosisArray = np.sort(absKurtosisArray)[::-1]

    if config['generateGraphs']:
        # plot adj rand index score
        plt.plot(range(icaTransformedX.shape[1]), sortedAbsKurtosisArray, marker='o')
        plt.title("Abs Kurtosis Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Sorted Component Order')
        plt.ylabel('Absolute Kurtosis')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_Kurtosis.png'))
        plt.show()

    # ICA reconstruction code copied from https://www.kaggle.com/ericlikedata/reconstruct-error-of-pca
    error_record = []
    numComponentsRange = range(1, len(ica.components_))
    for i in numComponentsRange:
        ica = FastICA(n_components=i, random_state=randomState)
        ica2_results = ica.fit_transform(X)
        ica2_proj_back = ica.inverse_transform(ica2_results)
        total_loss = LA.norm((X - ica2_proj_back), None)
        error_record.append(total_loss)

    # plot reconstruction error
    if config['generateGraphs']:
        plt.plot(numComponentsRange, error_record, marker='o')
        plt.title("Reconstruction Error Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Number of components')
        plt.ylabel('Error')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_ReconstructionError.png'))
        plt.show()

    algResultsDict['transformedX'] = icaTransformedX
    print(icaTransformedX.shape)
    return icaTransformedX


def runICA(config, X, datasetName, datasetResultsDict, datasetDict, randomState):
    print("Running " + config['name'] + " on " + datasetName)
    algResultsDict = {}
    algName = config['name']
    datasetResultsDict[algName] = algResultsDict

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # load dataset/algorithm specific config information (if any)
    numComponentsMin = config['icaNumComponentsMin']
    numComponentsMax = config['icaNumComponentsMax']
    numComponentsRange = range(numComponentsMin, numComponentsMax + 1)
    maxIterations = config['icaMaxIterations']
    tolerance = config['icaTolerance']
    gFunction = config['icaGFunction']

    meanAbsKurtosisArray = []
    reconstructionErrorArray = []
    for numComponents in numComponentsRange:
        ica = FastICA(n_components=numComponents, random_state=randomState, max_iter=maxIterations, tol=tolerance, fun=gFunction, whiten=True)
        icaTransformedX = ica.fit_transform(X)
        # kurtosis code copied from https://programtalk.com/python-examples/scipy.stats.kurtosis/
        kurtosisArray = stats.kurtosis(icaTransformedX)
        absKurtosisArray = np.absolute(kurtosisArray)
        meanAbsKurtosisArray.append(np.mean(absKurtosisArray))
        # ICA reconstruction code copied from https://www.kaggle.com/ericlikedata/reconstruct-error-of-pca
        inverseTransform = ica.inverse_transform(icaTransformedX)
        reconstructionError = LA.norm((X - inverseTransform), None)
        reconstructionErrorArray.append(reconstructionError)


    if config['generateGraphs']:
        # plot mean kurtosis/numComponents
        plt.plot(numComponentsRange, meanAbsKurtosisArray, marker='o')
        plt.title("Mean Abs Kurtosis Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Number of Components')
        plt.ylabel('Mean Absolute Kurtosis')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_MeanAbsKurtosis.png'))
        plt.show()

        # plot mean reconstruction error
        plt.plot(numComponentsRange, reconstructionErrorArray, marker='o')
        plt.title("Reconstruction Error Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Number of components')
        plt.ylabel('Error')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_ReconstructionError.png'))
        plt.show()

    algResultsDict['transformedX'] = icaTransformedX
    print(icaTransformedX.shape)
    return icaTransformedX


def runPCA(config, X, datasetName, datasetResultsDict, datasetDict, randomState):
    print("Running " + config['name'] + " on " + datasetName)
    algResultsDict = {}
    algName = config['name']
    datasetResultsDict[algName] = algResultsDict

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # load dataset/algorithm specific config information (if any)
    numComponents = None
    if 'pcaNumComponents' in config:
        numComponents = config['pcaNumComponents']

    pca = PCA(n_components=numComponents,  random_state=randomState, whiten=True).fit(X)
    pcaTransformedX = pca.transform(X)
    explainedVarianceRatioArray = pca.explained_variance_ratio_
    algResultsDict['transformedX'] = pcaTransformedX
    algResultsDict['explainedVarianceRatioArray'] = explainedVarianceRatioArray
    cumulativeExplainedVarianceRatioArray = np.cumsum(explainedVarianceRatioArray)
    algResultsDict['cumulativeExplainedVarianceRatioArray'] = cumulativeExplainedVarianceRatioArray
    plt.plot(cumulativeExplainedVarianceRatioArray)
    plt.title("Cumulative Explained Variance for " + datasetName)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.grid()

    if config['generateGraphs']:
        plt.savefig(path.join(resultsDirectory, config['name'] + 'CumExplainedVariance.png'))
        plt.show()
    plt.close('all')

    # PCA reconstruction code copied from https://www.kaggle.com/ericlikedata/reconstruct-error-of-pca
    error_record = []
    numComponentsRange = range(1, len(pca.components_))
    for i in numComponentsRange:
        pca = PCA(n_components=i, random_state=randomState)
        pca2_results = pca.fit_transform(X)
        pca2_proj_back = pca.inverse_transform(pca2_results)
        total_loss = LA.norm((X - pca2_proj_back), None)
        error_record.append(total_loss)

    # plot reconstruction error
    if config['generateGraphs']:
        plt.plot(numComponentsRange, error_record, marker='o')
        plt.title("Reconstruction Error Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Number of components')
        plt.ylabel('Error')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_ReconstructionError.png'))
        plt.show()

    plt.close('all')
    #pca = PCA(n_components=numComponents, random_state=randomState)
    #pca.fit(X)
    #pcaScore = pca.score(X)
    #pcaScoresArray.append(pcaScore)
    #print("For n_components =", numComponents,
    #      "The average score is : ", pcaScore)

    return pcaTransformedX


###############################################################
##################### Clusterting Methods #####################
###############################################################
def runEM(config, X, Y, datasetName, datasetResultsDict, datasetDict, randomState):
    print("Running " + config['name'] + " on " + datasetName)
    algResultsDict = {}
    algName = config['name']
    datasetResultsDict[algName] = algResultsDict

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # load dataset/algorithm specific config information (if any)
    numComponentsMin = config['emNumComponentsMin']
    numComponentsMax = config['emNumComponentsMax']
    numComponentsRange = range(numComponentsMin, numComponentsMax + 1)
    covarianceTypes = config['emCovarianceTypes']

    # the following code is borrowed (and modified) from
    #     http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    lowestBIC = np.infty
    bic = []
    for covarianceType in covarianceTypes:
        adjustedRandIndexScoreArray = []
        silhouetteAvgArray = []
        for numComponents in numComponentsRange:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=numComponents,
                                          covariance_type=covarianceType,
                                          random_state=randomState)
            gmm.fit(X)
            emLabels = gmm.predict(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowestBIC:
                lowestBIC = bic[-1]
                best_gmm = gmm

            # get adjusted rand index score
            adjustedRandIndexScore = adjusted_rand_score(emLabels, Y)
            adjustedRandIndexScoreArray.append(adjustedRandIndexScore)
            print("adjRandIndexScore for " + str(numComponents) + ": " + str(adjustedRandIndexScore))

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouetteAvg = silhouette_score(X, emLabels)
            silhouetteAvgArray.append(silhouetteAvg)
            # print("For n_clusters =", numClusters,
            #      "The average silhouette_score is :", silhouetteAvg)

            # Compute the silhouette scores for each sample
            sampleSilhouetteValues = silhouette_samples(X, emLabels)

            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (numComponents + 1) * 10])

            y_lower = 10
            for i in range(numComponents):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sampleSilhouetteValues[emLabels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / numComponents)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouetteAvg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(emLabels.astype(float) / numComponents)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            #centers = kMeansClusterer.cluster_centers_
            # Draw white circles at cluster centers
            #ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            #            c="white", alpha=1, s=200, edgecolor='k')
            #
            #for i, c in enumerate(centers):
            #    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
            #                s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for EM clustering on " + datasetName +
                          " with n_clusters = %d" % numComponents),
                         fontsize=14, fontweight='bold')

            if config['generateGraphs']:
                plt.savefig(path.join(resultsDirectory, config['name'] + '_' + covarianceType + '_' + str(numComponents) + '_Silhouette.png'))
                plt.show()
                plt.close('all')

        if config['generateGraphs']:
            # plot adj rand index score
            plt.plot(numComponentsRange, adjustedRandIndexScoreArray, marker='o')
            plt.title("Adjusted Rand Index Score Analysis of " + algName + " for " + datasetName)
            plt.xlabel('Number of clusters')
            plt.ylabel('Adjusted Rand Index Score')
            plt.savefig(path.join(resultsDirectory, config['name'] + '_' + covarianceType + '_AdjRandIndexScore.png'))
            plt.show()
            # plot the avg silhouette scores
            plt.plot(numComponentsRange, silhouetteAvgArray, marker='o')
            plt.title("Silhouette Avg Analysis of  " + algName + " for " + datasetName)
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette Avg')
            plt.savefig(path.join(resultsDirectory, config['name'] + '_' + covarianceType + '_SilhouetteAvg.png'))
            plt.show()



    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(covarianceTypes, color_iter)):
        xpos = np.array(numComponentsRange) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(numComponentsRange):
                                      (i + 1) * len(numComponentsRange)],
                            width=.2, color=color))
    plt.xticks(numComponentsRange)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model for ' + datasetName)
    xpos = np.mod(bic.argmin(), len(numComponentsRange)) + .65 + \
           .2 * np.floor(bic.argmin() / len(numComponentsRange))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], covarianceTypes)

    # Plot the winner
    #splot = plt.subplot(2, 1, 2)
    #Y_ = clf.predict(X)
    #for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
    #    v, w = linalg.eigh(cov)
    #    if not np.any(Y_ == i):
    #        continue
    #    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
#
        # Plot an ellipse to show the Gaussian component
    #    angle = np.arctan2(w[0][1], w[0][0])
    #    angle = 180. * angle / np.pi  # convert to degrees
    #    v = 2. * np.sqrt(2.) * np.sqrt(v)
    #    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    #    ell.set_clip_box(splot.bbox)
    #    ell.set_alpha(.5)
    #    splot.add_artist(ell)

    #plt.xticks(())
    #plt.yticks(())
    #plt.title('Selected GMM: full model, 2 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)

    if config['generateGraphs']:
        plt.savefig(path.join(resultsDirectory, config['name'] + 'BIC.png'))
        plt.show()
    plt.close('all')

    return np.append(X, emLabels.T.reshape(X.shape[0],1), axis=1)

def runKMeans(config, X, Y, datasetName, datasetResultsDict, datasetDict, randomState):
    print("Running " + config['name'] + " on " + datasetName)
    algResultsDict = {}
    algName = config['name']
    datasetResultsDict[algName] = algResultsDict

    algResultsDirectory = os.path.join(datasetDict['resultsDirectory'],
                                                algName)
    if not os.path.isdir(algResultsDirectory):
        os.makedirs(algResultsDirectory)
    resultsDirectory = algResultsDirectory

    # load dataset/algorithm specific config information (if any)
    numClustersMin = config['kMeansNumClustersMin']
    numClustersMax = config['kMeansNumClustersMax']
    numClustersRange = range(numClustersMin, numClustersMax + 1)

    clusterSizeArray = []
    silhouetteAvgArray = []
    distortionsArray = []
    clusterCentersArray = []
    vMeasureScoreArray = []
    adjustedRandIndexScoreArray = []
    # the following code is borrowed (and modified) from
    #     http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    for numClusters in numClustersRange:
        clusterSizeArray.append(numClusters)

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (numClusters + 1) * 10])

        kMeansClusterer = cluster.KMeans(n_clusters=numClusters, random_state=randomState, init='k-means++', n_init=10, max_iter=300)
        km = kMeansClusterer.fit(X)
        kMeansLabels = km.labels_
        distortionsArray.append(km.inertia_)
        clusterCenters = km.cluster_centers_
        clusterCentersArray.append(clusterCenters)

        vMeasureScore = v_measure_score(Y, kMeansLabels)
        vMeasureScoreArray.append(vMeasureScore)
        print("vMeasureScore for " + str(numClusters) + ": " + str(vMeasureScore))

        adjustedRandIndexScore = adjusted_rand_score(kMeansLabels, Y)
        adjustedRandIndexScoreArray.append(adjustedRandIndexScore)
        print("adjRandIndexScore for " + str(numClusters) + ": " + str(adjustedRandIndexScore))

        if len(datasetDict['outputClassNames']) == 2 and numClusters == 2:
            f1Score = f1_score(Y, kMeansLabels)
            print("f1Score for " + str(numClusters) + ": " + str(f1Score))

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouetteAvg = silhouette_score(X, kMeansLabels)
        silhouetteAvgArray.append(silhouetteAvg)
        #print("For n_clusters =", numClusters,
        #      "The average silhouette_score is :", silhouetteAvg)

        # Compute the silhouette scores for each sample
        sampleSilhouetteValues = silhouette_samples(X, kMeansLabels)

        y_lower = 10
        for i in range(numClusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sampleSilhouetteValues[kMeansLabels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / numClusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouetteAvg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(kMeansLabels.astype(float) / numClusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = kMeansClusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on " + datasetName +
                      " with n_clusters = %d" % numClusters),
                     fontsize=14, fontweight='bold')

        if config['generateGraphs']:
            # save the silhouette graph for this # of clusters
            plt.savefig(path.join(resultsDirectory, config['name'] + '_' + str(numClusters) + '_Silhouette.png'))
            plt.show()
            # save the cluster centers for this # of clusters
            f = open(path.join(resultsDirectory, config['name'] + '_' + str(numClusters) + '_ClusterCenters.txt'), "w+")
            f.write(str(clusterCenters))
            f.close()

            if numClusters == len(datasetDict['outputClassNames']):
                confusionMatrix = confusion_matrix(Y, kMeansLabels)
                plt.figure()
                Utils.plot_confusion_matrix(confusionMatrix, datasetDict['outputClassNames'], normalize=True)
                plt.tight_layout()
                # another way of doing it
                #plt.gcf().subplots_adjust(bottom=0.15)
                plt.savefig(path.join(resultsDirectory, config['name'] + '_' + str(numClusters) + '_NormalizedConfusion.png'))
                plt.show()
                plt.figure()
                Utils.plot_confusion_matrix(confusionMatrix, datasetDict['outputClassNames'], normalize=False)
                plt.tight_layout()
                # another way of doing it
                #plt.gcf().subplots_adjust(bottom=0.15)
                plt.savefig(path.join(resultsDirectory, config['name'] + '_' + str(numClusters) + '_NonNormalizedConfusion.png'))
                plt.show()

        plt.close('all')

    # plot distortion
    if config['generateGraphs']:
        # plot distortions
        plt.plot(numClustersRange, distortionsArray, marker='o')
        plt.title("Distortion Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_Distortion.png'))
        plt.show()
        #plt.close('all')

        # plot silhouetteAvg
        plt.plot(numClustersRange, silhouetteAvgArray, marker='o')
        plt.title("Silhouette Avg Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Avg')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_SilhouetteAvg.png'))
        plt.show()
        #plt.close('all')

        # plot vmeasurescore
        plt.plot(numClustersRange, vMeasureScoreArray, marker='o')
        plt.title("V-Measure Score Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Number of clusters')
        plt.ylabel('V-Measure Score')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_VMeasure.png'))
        plt.show()

        # plot adj rand index score
        plt.plot(numClustersRange, adjustedRandIndexScoreArray, marker='o')
        plt.title("Adjusted Rand Index Score Analysis of " + algName + " for " + datasetName)
        plt.xlabel('Number of clusters')
        plt.ylabel('Adjusted Rand Index Score')
        plt.savefig(path.join(resultsDirectory, config['name'] + '_AdjRandIndexScore.png'))
        plt.show()

    plt.close('all')

    kMeansResultDict = {}
    datasetResultsDict[algName] = kMeansResultDict

    kMeansResultDict['clusterSizeArray'] = clusterSizeArray
    kMeansResultDict['silhouetteAvgArray'] = silhouetteAvgArray

    return np.append(X, kMeansLabels.T.reshape(X.shape[0],1), axis=1)


###############################################################
##################### Main Method #############################
###############################################################
def main():
    resultsDict = {}

    # create the output directory, dictionary and log
    experimentResultsDirectory = os.path.join(os.getcwd(),
                                                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(experimentResultsDirectory)

    # load the configuration json file
    with open('./Configuration.json') as f:
        configJSON = json.load(f)

    # copy config file data to results directory
    with open(path.join(experimentResultsDirectory, 'config.json'), 'w') as f:
        f.write(json.dumps(configJSON))


    # load configuration data
    version = configJSON['version']
    randomState = configJSON['randomState']

    np.random.seed(randomState)

    # initialize the datasets
    datasetsDict = {}
    datasetConfigs = configJSON['datasets']
    for datasetConfig in datasetConfigs:
        if not datasetConfig['enabled']:
            continue
        datasetDict = {}
        datasetsDict[datasetConfig['name']] = datasetDict
        datasetDict['config'] = datasetConfig
        datasetDict['name'] = datasetConfig['name']
        initializeDataset(datasetDict, randomState)

    # now process each dataset
    for datasetName, datasetDict in datasetsDict.items():
        print("******* Processing " + datasetName + "...")
        datasetResultsDirectory = os.path.join(experimentResultsDirectory, datasetName)
        os.makedirs(datasetResultsDirectory)

        datasetDict['resultsDirectory'] = datasetResultsDirectory
        datasetResultDict = {}
        datasetDict['results'] = datasetResultDict

        datasetResultDict['version'] = version
        datasetResultDict['randomState'] = randomState

        X = datasetDict['X']
        Y = datasetDict['Y']
        XTrainUnmodified, XTestUnmodified, YTrainUnmodified, YTestUnmodified \
            = train_test_split(X, Y,
                               train_size=datasetDict['trainPercentage'],
                               test_size=1.0 -datasetDict['trainPercentage'],
                               random_state=randomState,
                               shuffle=datasetDict['shuffle'])

        datasetDict['XTrainUnmodified'] = XTrainUnmodified
        datasetDict['XTestUnmodified'] = XTestUnmodified
        datasetDict['YTrainUnmodified'] = YTrainUnmodified
        datasetDict['YTestUnmodified'] = YTestUnmodified


        ################################### do clustering ###################################
        ########### do kmeans
        # get kmeans config for this dataset
        kMeansConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'K-Means Clustering':
                kMeansConfig = algorithmSetting
                break

        if kMeansConfig is None:
            print("ERROR: Missing kmeans config for " + datasetName)
            sys.exit()

        if kMeansConfig['enabled']:
            runKMeans(kMeansConfig, datasetDict['X'], datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        kMeansConfig = None

        ########### do expectation maximization
        # get expectation maximization config for this dataset
        emConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'Expectation Maximization':
                emConfig = algorithmSetting
                break

        if emConfig is None:
            print("ERROR: Missing em config for " + datasetName)
            sys.exit()

        if emConfig['enabled']:
            runEM(emConfig, datasetDict['X'], datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        emConfig = None

        ################################### do dimensionality reduction #####################
        ########### do PCA
        # get PCA config for this dataset
        pcaConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'PCA':
                pcaConfig = algorithmSetting
                break

        if pcaConfig is None:
            print("ERROR: Missing pca config for " + datasetName)
            sys.exit()

        if pcaConfig['enabled']:
            runPCA(pcaConfig, datasetDict['X'], datasetName, datasetResultDict, datasetDict, randomState)

        pcaConfig = None

        ########### do ICA
        # get ICA config for this dataset
        icaConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'ICA':
                icaConfig = algorithmSetting
                break

        if icaConfig['enabled']:
            runICA(icaConfig, datasetDict['X'], datasetName, datasetResultDict, datasetDict, randomState)

        icaConfig = None

        ########### do Randomized Projections
        # get Randomized Projections config for this dataset
        randomizedProjectionsConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'Randomized Projection':
                randomizedProjectionsConfig = algorithmSetting
                break

        if randomizedProjectionsConfig['enabled']:
            runRandomizedProjections(randomizedProjectionsConfig, datasetDict['X'], datasetDict, randomState)

        randomizedProjectionsConfig = None

        ########### do LDA
        # get LDA config for this dataset
        ldaConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'LDA':
                ldaConfig = algorithmSetting
                break

        if ldaConfig['enabled']:
            runLDA(ldaConfig, datasetDict['XTrainUnmodified'], datasetDict['YTrainUnmodified'],
                   datasetDict['XTestUnmodified'], datasetDict['YTestUnmodified'],
                   datasetName, datasetResultDict,
                   datasetDict, randomState)

        ldaConfig = None

        ########### do RandomForestRegressor
        randomForestRegressorConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'Random Forest Regression':
                randomForestRegressorConfig = algorithmSetting
                break

        if randomForestRegressorConfig['enabled']:
            runRandomForestRegressor(randomForestRegressorConfig, datasetDict['X'], datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        randomForestRegressorConfig = None

        ################################### do clustering after dimensionality reduction ###################################
        ########### do kmeans
        ########### do kmeans with PCA
        # get kmeans config for this dataset/dimensionality reduction
        kMeansWithPCAConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'K-Means Clustering wPCA':
                kMeansWithPCAConfig = algorithmSetting
                break

        if kMeansWithPCAConfig is None:
            print("ERROR: Missing 'K-Means Clustering wPCA' config for " + datasetName)
            sys.exit()

        if kMeansWithPCAConfig['enabled']:
            pcaTransformedX = runPCA(kMeansWithPCAConfig, datasetDict['X'], datasetName, datasetResultDict, datasetDict, randomState)
            runKMeans(kMeansWithPCAConfig, pcaTransformedX, datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        kMeansWithPCAConfig = None

        ########### do kmeans with ICA
        # get kmeans config for this dataset/dimensionality reduction
        kMeansWithICAConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'K-Means Clustering wICA':
                kMeansWithICAConfig = algorithmSetting
                break

        if kMeansWithICAConfig is None:
            print("ERROR: Missing K-Means Clustering wICA config for " + datasetName)
            sys.exit()

        if kMeansWithICAConfig is not None and kMeansWithICAConfig['enabled']:
            icaTransformedX = runICA(kMeansWithICAConfig, datasetDict['X'], datasetName, datasetResultDict, datasetDict, randomState)
            runKMeans(kMeansWithICAConfig, icaTransformedX, datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        kMeansWithICAConfig = None

        ########### do kmeans with RP
        # get kmeans config for this dataset/dimensionality reduction
        kMeansWithRPConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'K-Means Clustering wRP':
                kMeansWithRPConfig = algorithmSetting
                break

        if kMeansWithRPConfig is None:
            print("ERROR: Missing K-Means Clustering wRC config for " + datasetName)
            sys.exit()

        if kMeansWithRPConfig is not None and kMeansWithRPConfig['enabled']:
            rpTransformedX = runRandomizedProjections(kMeansWithRPConfig, datasetDict['X'], datasetDict, randomState)
            runKMeans(kMeansWithRPConfig, rpTransformedX, datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        kMeansWithRPConfig = None

        ########### do kmeans with RandomForestRegresser
        # get kmeans config for this dataset/dimensionality reduction
        kMeansWithRFConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'K-Means Clustering wRF':
                kMeansWithRFConfig = algorithmSetting
                break

        if kMeansWithRFConfig is None:
            print("ERROR: Missing K-Means Clustering wRF config for " + datasetName)
            sys.exit()

        if kMeansWithRFConfig is not None and kMeansWithRFConfig['enabled']:
            rfTransformedX = runRandomForestRegressor(kMeansWithRFConfig, datasetDict['X'], datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)
            runKMeans(kMeansWithRFConfig, rfTransformedX, datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        kMeansWithRFConfig = None

        ########### do expectation maximization after dimensionality reduction ###################################
        ########### do expectation maximization with PCA
        # get EM config for this dataset/dimensionality reduction
        emWithPCAConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'EM Clustering wPCA':
                emWithPCAConfig = algorithmSetting
                break

        if emWithPCAConfig is None:
            print("ERROR: Missing 'EM Clustering wPCA' config for " + datasetName)
            sys.exit()

        if emWithPCAConfig['enabled']:
            pcaTransformedX = runPCA(emWithPCAConfig, datasetDict['X'], datasetName, datasetResultDict, datasetDict, randomState)
            runEM(emWithPCAConfig, pcaTransformedX, datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        emWithPCAConfig = None

        ########### do expectation maximization with ICA
        # get EM config for this dataset/dimensionality reduction
        emWithICAConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'EM Clustering wICA':
                emWithICAConfig = algorithmSetting
                break

        if emWithICAConfig is None:
            print("ERROR: Missing 'EM Clustering wICA' config for " + datasetName)
            sys.exit()

        if emWithICAConfig['enabled']:
            icaTransformedX = runICA(emWithICAConfig, datasetDict['X'], datasetName, datasetResultDict, datasetDict, randomState)
            runEM(emWithICAConfig, icaTransformedX, datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        emWithICAConfig = None

        ########### do expectation maximization with RP
        # get EM config for this dataset/dimensionality reduction
        emWithRPConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'EM Clustering wRP':
                emWithRPConfig = algorithmSetting
                break

        if emWithRPConfig is None:
            print("ERROR: Missing EM Clustering wRC config for " + datasetName)
            sys.exit()

        if emWithRPConfig is not None and emWithRPConfig['enabled']:
            rpTransformedX = runRandomizedProjections(emWithRPConfig, datasetDict['X'], datasetDict, randomState)
            runEM(emWithRPConfig, rpTransformedX, datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

        emWithRPConfig = None

        ########### do kmeans with RandomForestRegresser
        # get EM config for this dataset/dimensionality reduction
        emWithRFConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'EM Clustering wRF':
                emWithRFConfig = algorithmSetting
                break

        if emWithRFConfig is None:
            print("ERROR: Missing EM Clustering wRF config for " + datasetName)
            sys.exit()

        if emWithRFConfig is not None and emWithRFConfig['enabled']:
            rfTransformedX = runRandomForestRegressor(emWithRFConfig, datasetDict['X'], datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)
            runEM(emWithRFConfig, rfTransformedX, datasetDict['Y'], datasetName, datasetResultDict, datasetDict, randomState)

            emWithRFConfig = None


        ########### do ANN alone
        # get ANN alone config for this dataset/dimensionality reduction
        annConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'ANN':
                annConfig = algorithmSetting
                break

        if annConfig is None:
            print("ERROR: Missing ANN config for " + datasetName)
            #sys.exit()

        if annConfig is not None and annConfig['enabled']:
            runANN(annConfig, datasetDict['XTrainUnmodified'], datasetDict['YTrainUnmodified'],
                   datasetDict['XTestUnmodified'], datasetDict['YTestUnmodified'],
                   datasetName, datasetDict, datasetResultDict, randomState)

        annConfig = None

        ################################### do ANN after dimensionality reduction ###################################
        ####################  Each call to ANN has to have its training, validation and test sets modified by the ############
        ####################  dim reduction algorithms!  The hyperparameters for these shouldn't be chosen by cheating #######
        ####################  (i.e. by looking at test data) because, in the end, we are still doing supervised learning #####
        ########### do PCA then ANN
        # get PCA then ANN config for this dataset/dimensionality reduction
        pcaANNConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'ANN wPCA':
                pcaANNConfig = algorithmSetting
                break

        if pcaANNConfig is None:
            print("ERROR: Missing PCA ANN config for " + datasetName)
            sys.exit()

        if pcaANNConfig is not None and pcaANNConfig['enabled']:
            # run PCA twice, once for training data, once for test
            pcaTransformedXTrain = runPCA(pcaANNConfig, datasetDict['XTrainUnmodified'], datasetName, datasetResultDict, datasetDict,
                                     randomState)
            pcaTransformedXTest  = runPCA(pcaANNConfig, datasetDict['XTestUnmodified'],  datasetName, datasetResultDict, datasetDict,
                                     randomState)
            runANN(pcaANNConfig, pcaTransformedXTrain, datasetDict['YTrainUnmodified'],
                   pcaTransformedXTest, datasetDict['YTestUnmodified'],
                   datasetName, datasetDict, datasetResultDict, randomState)

        pcaANNConfig = None

        ########### do ICA then ANN
        icaANNConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'ANN wICA':
                icaANNConfig = algorithmSetting
                break

        if icaANNConfig is None:
            print("ERROR: Missing ICA ANN config for " + datasetName)
            #sys.exit()

        if icaANNConfig is not None and icaANNConfig['enabled']:
            # run ICA twice, once for training data, once for test
            icaTransformedXTrain = runICA(icaANNConfig, datasetDict['XTrainUnmodified'], datasetName, datasetResultDict, datasetDict,
                                     randomState)
            icaTransformedXTest = runICA(icaANNConfig, datasetDict['XTestUnmodified'], datasetName, datasetResultDict, datasetDict,
                                     randomState)
            runANN(icaANNConfig, icaTransformedXTrain, datasetDict['YTrainUnmodified'],
                   icaTransformedXTest, datasetDict['YTestUnmodified'],
                   datasetName, datasetDict, datasetResultDict, randomState)

        icaANNConfig = None

        ########### do RP then ANN
        rpANNConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'ANN wRP':
                rpANNConfig = algorithmSetting
                break

        if rpANNConfig is None:
            print("ERROR: Missing RP ANN config for " + datasetName)
            #sys.exit()

        if rpANNConfig is not None and rpANNConfig['enabled']:
            rpTransformedXTrain = runRandomizedProjections(rpANNConfig, datasetDict['XTrainUnmodified'], datasetDict, randomState)
            rpTransformedXTest = runRandomizedProjections(rpANNConfig, datasetDict['XTestUnmodified'], datasetDict, randomState)
            runANN(rpANNConfig, rpTransformedXTrain, datasetDict['YTrainUnmodified'],
                   rpTransformedXTest, datasetDict['YTestUnmodified'],
                   datasetName, datasetDict, datasetResultDict, randomState)

        rpANNConfig = None

        ########### do LDA then ANN
        ldaANNConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'ANN wLDA':
                ldaANNConfig = algorithmSetting
                break

        if ldaANNConfig is None:
            print("ERROR: Missing LDA ANN config for " + datasetName)
            #sys.exit()

        if ldaANNConfig is not None and ldaANNConfig['enabled']:
            runANN(ldaANNConfig, datasetDict['XTrainUnmodified'], datasetDict['YTrainUnmodified'],
                   datasetDict['XTestUnmodified'], datasetDict['YTestUnmodified'],
                   datasetName, datasetDict, datasetResultDict, randomState)

        ldaANNConfig = None

        ########### do RandomForestRegressor then ANN
        rfANNConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'ANN wRF':
                rfANNConfig = algorithmSetting
                break

        if rfANNConfig is None:
            print("ERROR: Missing RF ANN config for " + datasetName)
            #sys.exit()

        if rfANNConfig is not None and rfANNConfig['enabled']:
            rfTransformedXTrain = runRandomForestRegressor(rfANNConfig, datasetDict['XTrainUnmodified'], datasetDict['YTrainUnmodified'], datasetName, datasetResultDict, datasetDict, randomState)
            rfTransformedXTest  = runRandomForestRegressor(rfANNConfig, datasetDict['XTestUnmodified'], datasetDict['YTestUnmodified'], datasetName, datasetResultDict, datasetDict, randomState)
            runANN(rfANNConfig, rfTransformedXTrain, datasetDict['YTrainUnmodified'],
                   rfTransformedXTest, datasetDict['YTestUnmodified'],
                   datasetName, datasetDict, datasetResultDict, randomState)

        rfANNConfig = None

        ################################### do ANN after clustering to create new feature ###################################
        ####################  Each call to ANN has to have its training, validation and test sets modified by the ############
        ####################  clustering algorithms!  The hyperparameters for these shouldn't be chosen by cheating    #######
        ####################  (i.e. by looking at test data) because, in the end, we are still doing supervised learning #####
        ########### do kmeans
        kmeansANNConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'ANN wKMeans':
                kmeansANNConfig = algorithmSetting
                break

        if kmeansANNConfig is None:
            print("ERROR: Missing KMeans ANN config for " + datasetName)
            #sys.exit()

        if kmeansANNConfig is not None and kmeansANNConfig['enabled']:
            kmTransformedXTrain = runKMeans(kmeansANNConfig, datasetDict['XTrainUnmodified'], datasetDict['YTrainUnmodified'], datasetName, datasetResultDict, datasetDict, randomState)
            kmTransformedXTest  = runKMeans(kmeansANNConfig, datasetDict['XTestUnmodified'], datasetDict['YTestUnmodified'], datasetName, datasetResultDict, datasetDict, randomState)
            runANN(kmeansANNConfig, kmTransformedXTrain, datasetDict['YTrainUnmodified'],
                   kmTransformedXTest, datasetDict['YTestUnmodified'],
                   datasetName, datasetDict, datasetResultDict, randomState)

        kmeansANNConfig = None

        ########### do expectation maximization
        emANNConfig = None
        for algorithmSetting in datasetDict['config']['algorithmSettings']:
            if algorithmSetting['name'] == 'ANN wEM':
                emANNConfig = algorithmSetting
                break

        if emANNConfig is None:
            print("ERROR: Missing EM ANN config for " + datasetName)
            #sys.exit()

        if emANNConfig is not None and emANNConfig['enabled']:
            emTransformedXTrain = runEM(emANNConfig, datasetDict['XTrainUnmodified'], datasetDict['YTrainUnmodified'], datasetName, datasetResultDict, datasetDict, randomState)
            emTransformedXTest = runEM(emANNConfig, datasetDict['XTestUnmodified'], datasetDict['YTestUnmodified'], datasetName, datasetResultDict, datasetDict, randomState)
            runANN(emANNConfig, emTransformedXTrain, datasetDict['YTrainUnmodified'],
                   emTransformedXTest, datasetDict['YTestUnmodified'],
                   datasetName, datasetDict, datasetResultDict, randomState)

        emANNConfig = None

    # this recursively converts all the non-dict values in the dict to strings
    # it's needed because json.dumps can't persist certain numpy numeric datatypes
    # and crashes and therefore doesn't write out the final file!
    Utils.convert_all_dict_vals_to_string(resultsDict)
    with open(path.join(experimentResultsDirectory, 'results.json'), 'w') as f:
        f.write(json.dumps(resultsDict))


if __name__ == "__main__":
    main()
