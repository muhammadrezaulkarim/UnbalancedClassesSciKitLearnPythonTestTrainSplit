#Written By: Muhammad Rezaul Karim
# Used API Links:
#1. http://scikit-learn.org/stable/modules/classes.html#
#2. http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html

#To run experiments in python:
#1. Download and install Anaconda: https://www.anaconda.com/download/
#It Install the necessary python libraries (e.g. scikit-learn, numpy, matplotlib etc.)
#2.Install imbalanced-learn python library seperately. Run the following command in the Anaconda command prompt:
# conda install -c glemaitre imbalanced-learn
#3. Open Anaconda Navigator. Launch 'jupyter notbook' to run python code


# This file is open for public re-use.
# With this file, you can perform machine learning based classification experiments for unbalanced data sets
# where you have a series of training data sets (training-dataset0, training-dataset1, training-dataset2 etc.) 
# and a series of test data sets (test-dataset0, test-dataset1, test-dataset2 etc.). 
# For each trial X (X=0,1,...,NUM_OF_RUNS), training-datasetX will be used for training purpose and test-datasetX will
# be used for testing

#The last column in each data set must contain the dependent (class) variable, while all the other columns
# are independent (predictor) variables. 

#General Steps for Building Classification Models with a given data set:
#Step 1: Clean your data set (Impute missing values, remove samples with missing values etc.). Cleaning actions
# are problem specific, not performed in this file. Use an already cleaned data set for experimentation
#Step 2: Balance your data set. This code is written to work with unbalanced data sets. Use one of the balancing methods
#listed below. These techniques have their own parameters which can be tuned. By default one is selected
#Step 3: Transform your features (normalize, standardize etc.) to different range of values (e.g. [0,1])
#Step 4: Perform feature selection (reduce the number of features, keep important features etc.)
# Use one of the feature selection methods listed below. By default one is selected
#Step 5: Optimize hyper-parameters (Tune parameters) with K-fold stratified cross-validation
#Step 6: Build the actual models and perfrom evaluations with the test sets

######Important#############
# For handling missing Values: use 'sklearn.preprocessing.Imputer'
# Use 'sklearn.preprocessing.OneHotEncoder', (OneHotEncoder encoding) to feed categorical predictors to linear models 
# and SVMs with the standard kernels.

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids

import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
import statistics

from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

    
 
#training data sets should be named like this: training-dataset0, training-dataset1, training-dataset2 etc.
#test data sets should be named like this: test-dataset0, test-dataset1, test-dataset2 etc.
#make sure that file paths are correct for:  'trainingDataSetFileName' and 'testDataSetFileName' (see below)

NUM_RUNS = 3 # The number of trials
NUM_JOBS=8  # The number of threads for parallelization. Each CPU core can handle 2 threads
 # seed used by the random number generator. Used for re-producability
    
fScoreList=[]
precScoreList=[]
recallScoreList=[]
class_names=[]
# Loop for each trial
for i in range(0,NUM_RUNS):
    print()
    print()
    print("###### Run " + str(i) + " has started #####")
    seed=NUM_RUNS  

    ########################### Load Data Set From File################################################################
    #Load time wise training data set (e.g. daily, weekly training data) with python panda read_csv method
    trainingDataSetFileName="C:/Users/r_kar/OneDrive/Pictures/Documents/training-dataset"+ str(i) + ".csv"
    dataset1=pd.read_csv(trainingDataSetFileName)
    numOfrows1, numOfColumns1=dataset1.shape
    print("Dimension of the training data set:", dataset1.shape)  # number of rows and columns in the data set

    #Load test data (e.g. daily, weekly test data) with python panda read_csv method
    testDataSetFileName="C:/Users/r_kar/OneDrive/Pictures/Documents/test-dataset"+ str(i) + ".csv"
    dataset2=pd.read_csv(testDataSetFileName)
    numOfrows2, numOfColumns2=dataset2.shape
    print("Dimension of the test data set:", dataset2.shape)  # number of rows and columns in the data set



    dataset_data_training=dataset1.iloc[ : ,0:numOfColumns1-1] #all predictor variable
    dataset_target_training=dataset1.iloc[ : ,numOfColumns1-1] # dependent variable. Assumption is that the last column contains 
    #the dependent variable

    dataset_data_test=dataset2.iloc[ : ,0:numOfColumns2-1] #all predictor variable
    dataset_target_test=dataset2.iloc[ : ,numOfColumns2-1] # dependent variable. Assumption is that the last column contains 
    #the dependent variable


    # LabelEncoder will convert string class names into numeric class names (e.g. 0,1,2 etc.)
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(list(dataset_target_training))
    convertedIntoClasses=labelEncoder.transform(list(dataset_target_training))
    
    encodedClasses=np.unique(np.array(convertedIntoClasses)) # unique labels in the converted labels
    print("New names for classes (training):",encodedClasses)
    print("Actual names for classes (training):",labelEncoder.inverse_transform(encodedClasses))
    print()
    # Use the newly encoded class names
    dataset_target_training=convertedIntoClasses
    class_names= np.array(labelEncoder.inverse_transform(encodedClasses)) # store the class mapping
    print("Count of Samples Per Class in unbalanced state (training):")  
    print(sorted(Counter(dataset_target_training).items())) #count of different classes

    # Now convert the test classes in the same way
    convertedIntoClasses1=labelEncoder.transform(list(dataset_target_test))
    encodedClasses1=np.unique(np.array(convertedIntoClasses1)) # unique labels in the converted labels
    dataset_target_test=convertedIntoClasses1
    #print(dataset_target_test)
 

    ###########################Resolve Class Imbalance (only one technique will be used)################################################################
    #Class Imbalance Handling 1 (An Under-sampling Technique):  
    #ClusterCentroids makes use of K-means to reduce the number of samples. Therefore, 
    #each class will be synthesized with the centroids of the K-means method instead of the original samples:)
    #imbalanceHandeler = ClusterCentroids(random_state=seed)
    #X_resampled_training, y_resampled_training = imbalanceHandeler.fit_sample(dataset_data_training, dataset_target_training)
    #print("Count of Samples Per Class in balanced state:")   
    #print(sorted(Counter(y_resampled_training).items()))
    #dataset_data_training = X_resampled_training
    #dataset_target_training = y_resampled_training

    ##Class Imbalance Handling 2 (An Under-sampling Technique): RandomUnderSampler is a fast and easy way to balance the data 
    #by randomly selecting a subset of data for the targeted classes:
    imbalanceHandeler  = RandomUnderSampler(random_state=seed)
    X_resampled_training, y_resampled_training = imbalanceHandeler.fit_sample(dataset_data_training, dataset_target_training)
    #print()
    print("Count of Samples Per Class in balanced state(training):")  
    print(sorted(Counter(y_resampled_training).items()))
    print()
    dataset_data_training = X_resampled_training
    dataset_target_training = y_resampled_training

    # Class Imbalance Handling 3 (An Under-sampling Technique): NearMiss method implements 3 different types of heuristic 
    # which can be selected with the parameter version (version=1, version=2, version=3):
    #imbalanceHandeler =  NearMiss(random_state=seed, version=1,n_jobs=NUM_JOBS)
    #X_resampled_training, y_resampled_training = imbalanceHandeler.fit_sample(dataset_data_training, dataset_target_training)
    #print("Count of Samples Per Class in balanced state:") 
    #print(sorted(Counter(y_resampled_training).items()))
    #dataset_data_training = X_resampled_training
    #dataset_target_training = y_resampled_training

    #Class Imbalance Handling 4 (An Over-sampling Technique) by SMOTE (Synthetic Minority-Over Sampling Technique)
    # k_neighbors: number of nearest neighbours to used to construct synthetic samples
    # n_jobs: The number of threads to open if possible
    #imbalanceHandeler =  SMOTE(random_state=seed, ratio='auto', kind='regular', k_neighbors=5, n_jobs=NUM_JOBS)
    #X_resampled_training, y_resampled_training = imbalanceHandeler.fit_sample(dataset_data_training, dataset_target_training)
    #print("Count of Samples Per Class in balanced state:") 
    #print(sorted(Counter(y_resampled_training).items()))
    #dataset_data_training = X_resampled_training
    #dataset_target_training = y_resampled_training

    #Class Imbalance Handling 5 (Combine over- and under-sampling using SMOTE and Tomek links). Perform over-sampling using SMOTE 
    #and cleaning using Tomek links. Tomek method performs under-sampling by removing Tomek’s links.
    # A Tomek’s link exist if the two samples are the nearest neighbors of each other
    #smoteObject=SMOTE(random_state=seed, ratio='auto', kind='regular', k_neighbors=5, n_jobs=NUM_JOBS)
    #tomekObject=TomekLinks(random_state=seed,ratio='auto', n_jobs=NUM_JOBS)
    #imbalanceHandeler =  SMOTETomek(random_state=seed,ratio='auto', smote=smoteObject,tomek=tomekObject)
    #X_resampled_training, y_resampled_training = imbalanceHandeler.fit_sample(dataset_data_training, dataset_target_training)
    #print("Count of Samples Per Class in balanced state:") 
    #print(sorted(Counter(y_resampled_training).items()))
    #dataset_data_training = X_resampled_training
    #dataset_target_training = y_resampled_training


    # Please do not comment out this lines.
    # This part is common for the above balancing methods. Executed after the balancing process.
    X_dataset_training,Y_dataset_training = dataset_data_training,dataset_target_training
    X_data_training, y_data_training = X_dataset_training[:, 0:len(X_dataset_training[0])], Y_dataset_training
    print("Before feature selection. Note the number of predictors (second value)") 
    print(X_data_training.shape)  #reduced data set number of rows and columns
    
    #Just copy the test data here. For test data, balancing not required. Direct prediction will be made
    X_data_test, y_data_test = dataset_data_test,dataset_target_test

    #####################Feature transformation (only one of the following methods need to be used)############################################
    #print("Feature values after transformation") 
    #Feature Transformation 1: Standardize features to 0 mean and unit variance
    #For training data
    #scaler = preprocessing.StandardScaler().fit(X_data_training)
    #X_data_transformed_training = scaler.transform(X_data_training)
    #X_data_training=X_data_transformed_training
    #For test data 
    #scaler = preprocessing.StandardScaler().fit(X_data_test)
    #X_data_transformed_test = scaler.transform(X_data_test)
    #X_data_test=X_data_transformed_test
    

    #Feature Transformation 2: transforms features by scaling each feature to a [0,1] range.
    #For training data
    scaler = preprocessing.MinMaxScaler().fit(X_data_training)
    X_data_transformed_training = scaler.transform(X_data_training)
    X_data_training=X_data_transformed_training
    #For test data
    scaler = preprocessing.MinMaxScaler().fit(X_data_test)
    X_data_transformed_test = scaler.transform(X_data_test)
    X_data_test=X_data_transformed_test
    
    

    #Feature Transformation 3: Normalize samples individually to unit norm.
    #For training data
    #scaler = preprocessing.Normalizer().fit(X_data_training)
    #X_data_transformed_training = scaler.transform(X_data_training)
    #X_data_training=X_data_transformed_training
    #For test data
    #scaler = preprocessing.Normalizer().fit(X_data_test)
    #X_data_transformed_test = scaler.transform(X_data_test)
    #X_data_test=X_data_transformed_test

    ################# Perform feature selection (only one of the following methods need to be used)########################################################
    print("After feature selection. Note the number of predictors for training set(second value)") 
    # Feature Selection Method 1: Random Forest Based feature selection
    clf = RandomForestClassifier(random_state=seed)
    clf = clf.fit(X_data_training,y_data_training)
    #print(clf.feature_importances_ ) 
    selector = SelectFromModel(clf, prefit=True)
    X_new_training = selector.transform(X_data_training)
    print(X_new_training.shape)  #reduced data set number of rows and columns
    X_data_training=X_new_training
 

    #Feature Selection Method 2: chi^2 test based top k feature selection
   # selector= SelectKBest(chi2, k=17)
   # X_new_training = selector.fit_transform(X_data_training, y_data_training)
   # X_data_training=X_new_training
   # print(X_new_training.shape)

    #Feature Selection Method 3: Recursive feature elimination with cross-validation
    #estimator = SVR(kernel="linear")
    #selector = RFECV(estimator, step=1, cv=5)
    #selector = selector.fit(X_data_training, y_data_training)
    #X_new_training=selector.transform(X_data_training)
    #X_data_training=X_new_training
    #print(X_new_training.shape)
    
    
    # Now select the data related to the filtered features from the test set 
    selectedFeatureIndexes=selector.get_support(indices=True) # get the integer indexes of the selected features
    X_data_test=np.array(X_data_test)
    X_data_test=X_data_test[:,np.array(selectedFeatureIndexes)] # array based indexing
    print("After feature selection. Note the number of predictors for test set(second value)") 
    print(X_data_test.shape)  #reduced data set number of rows and columns
    
    #################Tune parameters, build the actual models and perfrom evaluations #################


    #This is for multi class classification
    classifier = OneVsRestClassifier(RandomForestClassifier(random_state=seed))

    #Max depth and max_features need to be less than the selected features. So check
    #the number of selected features
    parameters_grid = {
       # "estimator__n_estimators": [10,20,50,100], # The number of trees in the forest.
        "estimator__n_estimators": [10,25,50,100,150], # The number of trees in the forest.
        "estimator__max_depth": [2,4,8,16],    # maximum depth of each tree
        "estimator__max_features": [2,4,8,16], # max features per random selection
       # "estimator__max_leaf_nodes":[2,3,5,7,10],  # max leaf nodes per tree. minimum 1
       # "estimator__min_samples_leaf":[2,5,7,10], # min # samples per leaf
    }


    #estimator: estimator object for GridSearchCV


    # Arrays to store scores
    grid_search_best_scores = np.zeros(NUM_RUNS)  # numpy arrays

    folds_for_grid_search = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
    # Parameter tuning with grid search and cross validation. Scoring function can be other measures like Area Under the ROC
    # curve. 
    #n_jobs=Number of jobs to run in parallel
    tuned_model = GridSearchCV(estimator=classifier, param_grid=parameters_grid, cv= folds_for_grid_search, scoring='f1_macro',
                               n_jobs=NUM_JOBS)     
    
    tuned_model.fit(X_data_training, y_data_training)
    grid_search_best_scores[i] = tuned_model.best_score_
    
   
    # print (tuned_model.best_score_)
    print()
    print("Best Selected Parameters:")
    print(tuned_model.best_params_)
   
    
    #Predict on test data. y_pred_test contains predictions for each sample
    y_pred_test = tuned_model.best_estimator_.predict(X_data_test)
   
    # print the prediction on test sets (if needed)
    #print( y_pred_test)

    #get class wise precision score and store the results in a list
    #Set 'average=None' to get class wise results
    precisionResult=precision_score(y_data_test,y_pred_test,average=None)
    precScoreList.append(precisionResult)
    
    #get class wise recall score  and store the results in a list
    recallResult=recall_score(y_data_test,y_pred_test,average=None)
    recallScoreList.append(recallResult)

    #get class wise f-measure score and store the results in a list
    fScoreResult=f1_score(y_data_test,y_pred_test, average=None)
    fScoreList.append(fScoreResult)
    
    print()
    print()
    print("***Scikit learn will set a metric (e.g. recall) value to zero and display a warning message ") 
    print("when no samples present for a particular class in the test set***")
#For loop ends here
#Print the results of the list thta contain the results  
#print(precScoreList)
#print(recallScoreList)
#print(fScoreList)



NUM_OF_CLASSES=len(precScoreList[0])  # automatically determine the number of classes 
print()  
print()  

for i in range(0,NUM_OF_CLASSES):
    print()
    print("Results for the class:: "+ class_names[i])
    print("####################################################")
    
    # store class wise precision, recall and F-Measure
    fScoreArray=array.array("d") #type of the array
    precScoreArray=array.array("d") 
    recallScoreArray=array.array("d") 
    
    #for the current class, extract precision, recall and F-Measure values for each run/trial 
    for j in range(0,len(precScoreList)):
        # Scikit learn will set the precision value to zero when no samples present for a class. We will ignore that result while computing average and variance
        if precScoreList[j][i] > 0.0:  
            precScoreArray.append(precScoreList[j][i])
        # Scikit learn will set the recall value to zero when no samples present for a class. We will ignore that result while computing average and variance
        if recallScoreList[j][i] > 0.0: 
            recallScoreArray.append(recallScoreList[j][i])
        # Scikit learn will set the F-measure value to zero when no samples present for a class. We will ignore that result while computing average and variance
        if fScoreList[j][i] > 0.0: 
            fScoreArray.append(fScoreList[j][i])
   
  #  print(fScoreArray)
  #  print(precScoreArray)
  #  print(recallScoreArray)
    
    print()    
    print("F1 Score (Average, Variance):", (statistics.mean(fScoreArray), statistics.variance(fScoreArray)))
    print("Precision Score (Average, Variance):",(statistics.mean(precScoreArray), statistics.variance(precScoreArray)))
    print("Recall Score (Average, Variance):", (statistics.mean(recallScoreArray), statistics.variance(recallScoreArray)))





