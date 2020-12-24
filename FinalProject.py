#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:55:14 2019

@author: deepesh
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as matplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential ,load_model
from numpy import sqrt
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  Activation ,MaxPool2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import  ZeroPadding2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras import losses
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix ,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
#from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# Citation : https://stackoverflow.com/questions/17626694/remove-python-userwarning/17626733
#Ignoring warning for Collinear
import warnings
warnings.filterwarnings("ignore")

#All plots that i have used is https://www.tutorialspoint.com/numpy/numpy_matplotlib.htm

#### Generic Methods used for Question1 ##################
#
#####################################################  

        
def createShuffleSplit(shuffleData,shufflelabel,splitSize,size):
    shuffle_shift_data = StratifiedShuffleSplit(n_splits=splitSize,train_size=0.8,test_size=size, random_state=0)    
    shuffle_shift_data.get_n_splits(shuffleData, shufflelabel)
    return shuffle_shift_data


def computeAlgoScenesBehind(classifierLables,classfiers,train_X_any,train_y_any,test_X_any,test_y_any):    
    classfiers.fit(train_X_any, train_y_any)
    allalgo_actual_pred = classfiers.predict(test_X_any)
    error_rates = computeErrorRate(classfiers,allalgo_actual_pred,test_y_any)
    accuracy_score = computeScoreofAllAlgo(classifierLables,test_y_any,allalgo_actual_pred)
    return allalgo_actual_pred,error_rates,accuracy_score;
              

def computeErrorRate(classfiers,allalgo_actual_pred,test_y_any):
    error_rate = np.mean(test_y_any != allalgo_actual_pred)
    return error_rate


def computeScoreofAllAlgo(algoLabel,ylables,allAlgopredcitions):
#    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    score = accuracy_score(ylables, allAlgopredcitions)
#    print(algoLabel + 'Acuuracy :  ',score)
    return score


def computeAlgoMagic(shullfeDataResponse,shuffleData,ylabels,questionType,Xdata_Testing,ylables_Testing):
    
    gaussian_error_rate = []
    quadratic_error_rate = []
    ldda_error_rate= []
    knn_error_rate_one = []
    knn_error_rate_five = []
    knn_error_rate_ten = []
    
    
    knn_accuracy_rate_rate_one = []
    knn_accuracy_rate_rate_five = []
    knn_accuracy_rate_rate_ten = []
    
    gaussian_accuracy_rate = []
    quadratic_accuracy_rate_rate = []
    ldda_accuracy_rate_rate= []

    for pointer_for_data_train, pointer_for_data_test in shullfeDataResponse.split(shuffleData, ylabels):

        train_X, test_X = shuffleData[pointer_for_data_train], shuffleData[pointer_for_data_test]
        train_y, test_y = ylabels[pointer_for_data_train],ylabels[pointer_for_data_test]
    
        
        lda_actual_pred ,error_rate,accuracyrate = computeAlgoScenesBehind(" LDA ",LDA(),train_X,train_y,test_X,test_y)
        ldda_error_rate.append(error_rate)
        ldda_accuracy_rate_rate.append(accuracyrate)
                
        gaussian_actual_pred ,error_rate,accuracyrate = computeAlgoScenesBehind(" Gaussian ",GaussianNB(),train_X,train_y,test_X,test_y)
        gaussian_error_rate.append(error_rate)
        gaussian_accuracy_rate.append(accuracyrate)
       
        quadratic_actual_pred ,error_rate,accuracyrate = computeAlgoScenesBehind("QDA ",QDA(),train_X,train_y,test_X,test_y)
        quadratic_error_rate.append(error_rate)
        quadratic_accuracy_rate_rate.append(accuracyrate)
        
        
        
        knn_actual_one  ,error_rate,accuracyrate = computeAlgoScenesBehind("KNN ONE ",KNeighborsClassifier(n_neighbors = 1, n_jobs=-1),train_X,train_y,test_X,test_y)
        knn_error_rate_one.append(error_rate)
        knn_accuracy_rate_rate_one.append(accuracyrate)
       
        knn_actual_two ,error_rate,accuracyrate = computeAlgoScenesBehind("KNN FIVE ",KNeighborsClassifier(n_neighbors = 5, n_jobs=-1),train_X,train_y,test_X,test_y)
        knn_error_rate_five.append(error_rate)
        knn_accuracy_rate_rate_five.append(accuracyrate)
       
        knn_actual_ten  ,error_rate,accuracyrate = computeAlgoScenesBehind("KNN TEN ",KNeighborsClassifier(n_neighbors = 10, n_jobs=-1),train_X,train_y,test_X,test_y)
        knn_error_rate_ten.append(error_rate)  
        knn_accuracy_rate_rate_ten.append(accuracyrate)
        
    
        
#        
    plotGraphErrororAccuracy("Error",questionType,ldda_error_rate,gaussian_error_rate,
                              quadratic_error_rate,knn_error_rate_one,
                              knn_error_rate_five,knn_error_rate_ten)
    
    plotGraphErrororAccuracy("Accuracy",questionType,ldda_accuracy_rate_rate,gaussian_accuracy_rate,
                              quadratic_accuracy_rate_rate,knn_accuracy_rate_rate_one,
                              knn_accuracy_rate_rate_five,knn_accuracy_rate_rate_ten)
                   
        
    
    print('LDA MEAN ERROR RATE',np.mean(ldda_error_rate))
    print('GAUSSIAN MEAN ERROR RATE ',np.mean(gaussian_error_rate))
    print('QUAD mean ERROR RATE',np.mean(quadratic_error_rate))
    
    print('LDA MEAN ACCURACY RATE',np.mean(ldda_accuracy_rate_rate))
    print('GAUSSIAN MEAN ACCURACY RATE ',np.mean(gaussian_accuracy_rate))
    print('QUAD mean ACCURACY RATE',np.mean(quadratic_accuracy_rate_rate))
    
    
    print('KNN ONE mean ERROR RATE',np.mean(knn_error_rate_one))
    print('KNN TWO mean ERROR RATE',np.mean(knn_error_rate_five))
    print('KNN TEN mean ERROR RATE ',np.mean(knn_error_rate_ten))
    
    
    print('KNN ONE mean ACCURACY RATE',np.mean(knn_accuracy_rate_rate_one))
    print('KNN TWO mean ACCURACY RATE',np.mean(knn_accuracy_rate_rate_five))
    print('KNN TEN mean ACCURACY RATE ',np.mean(knn_accuracy_rate_rate_ten))
  
               
    return 



#citation
#https://matplotlib.org/2.1.2/api/_as_gen/matplotlib.pyplot.plot.html
def plotGraphErrororAccuracy(plotType,questionType,lda_rate,gaussian_rate,
                              quadratic_rate,knn_rate_one,knn_rate_five,
                              knn_rate_ten):

    matplot.figure(figsize=(8,6))
    matplot.plot(range(0,5),lda_rate,color='blue', label='LDA '+ plotType + 'rate' , marker='o',
             markerfacecolor='red', markersize=5)
    matplot.plot(range(0,5),gaussian_rate,color='red', label='Gaussian '+ plotType + ' rate' , marker='o',
             markerfacecolor='red', markersize=5)
    matplot.plot(range(0,5),quadratic_rate,color='yellow', label='Quadratic '+ plotType + ' rate' , marker='o',
             markerfacecolor='red', markersize=5)
    matplot.plot(range(0,5),knn_rate_one,color='green', label='KNN  - 1 '+plotType , marker='o',
             markerfacecolor='red', markersize=5)
    matplot.plot(range(0,5),knn_rate_five,color='green', label='KNN  - 5 '+plotType , marker='o',
             markerfacecolor='blue', markersize=5)
    matplot.plot(range(0,5),knn_rate_ten,color='green', label='KNN  - 10 '+plotType , marker='o',
             markerfacecolor='black', markersize=5)
    if plotType == "Error" :
         matplot.title(' Question 1 : Testing Error Rate for LDA ,QDA ,Gaussian ,KNN 1,5,10')

    else :
         matplot.title('Question 3 : Testing Accuracy Rate for LDA ,QDA ,Gaussian ,KNN 1,5,10')
       
       
        
    matplot.xlabel('StratifiedShuffleSplit Tension Range')
    if plotType == "Error" :
        matplot.ylabel('Error Rate')
    else :
        matplot.ylabel('Accuracy Rate')
    
    matplot.legend(loc='best')
    matplot.show()
    return

#### End of Generic Methods used for Assignment One and Assignment 2 ##################

#####################################################   
    
#citation
#https://keras.io/layers/core/
#https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/


def getData():
    
    Xdata = np.load('Training_resized/resized_all.npy')
    ylables = np.load('Training_resized/resized_labels.npy')
    
    Xdata_Testing = np.load('Test_resized/resized_all.npy')
    ylables_Testing = np.load('Test_resized/resized_labels.npy')
    
    Xdata = np.reshape(Xdata,(len(Xdata), 2700))
    Xdata_Testing = np.reshape(Xdata_Testing,(len(Xdata_Testing),2700))

    return Xdata,ylables,Xdata_Testing,ylables_Testing


def Question_LDA_QDA_KKN_GAUSSIAN():
    
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()

    shullfeDataResponse=createShuffleSplit(Xdata,ylables,5,0.2)
    computeAlgoMagic(shullfeDataResponse,Xdata,ylables,"questionLQG",Xdata_Testing,ylables_Testing)   
    
    return

def QuestionPCA():
      
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()
    pca = PCA(n_components = 2)
    Xdata = pca.fit_transform(Xdata)
    Xdata_Testing = pca.transform(Xdata_Testing)
    explained_variance = pca.explained_variance_ratio_
      
    return



def QuestionOneCNN(QuestionType):
    
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()
    
    
    print('printing train_X shapes',np.shape(Xdata))
    print('printing test_X shapes',np.shape(ylables))
    print('printing y_train shapes',np.shape(Xdata_Testing))
    print('printing y_train shapes',np.shape(ylables_Testing))
    
    
    ylables_Testing = np.array(ylables_Testing)
    label_encoder = LabelEncoder()
    ylables_Testing = label_encoder.fit_transform(ylables_Testing)
      
    ylables_Testing = to_categorical(ylables_Testing)
    
    Xdata_Testing  = Xdata_Testing.reshape(Xdata_Testing.shape[0], 30, 30, 3)
    Xdata_Testing = Xdata_Testing.astype('float32')
           
    Xdata_Testing /=255
        
   
    stratifiedShuffleSplit = StratifiedKFold(n_splits=10, random_state=2)
    stratifiedShuffleSplit.get_n_splits(Xdata, ylables)
    for train_index, val_index in stratifiedShuffleSplit.split(Xdata,ylables): 

        training_x_data, testing_x_data = Xdata[train_index], Xdata[val_index] 
        training_y_data, testing_y_data = ylables[train_index], ylables[val_index]
        
        
        training_y_data = np.array(training_y_data)
        label_encoder = LabelEncoder()
        training_y_data = label_encoder.fit_transform(training_y_data)
        
        testing_y_data = np.array(testing_y_data)
        label_encoder = LabelEncoder()
        testing_y_data = label_encoder.fit_transform(testing_y_data)
        
        training_x_data = training_x_data.reshape(training_x_data.shape[0],30, 30, 3)
        testing_x_data = testing_x_data.reshape(testing_x_data.shape[0], 30, 30, 3)
        
  
        training_y_data = to_categorical(training_y_data)
        testing_y_data = to_categorical(testing_y_data)
      
       
        training_x_data = training_x_data.astype('float32')
        testing_x_data = testing_x_data.astype('float32')
       
            
        training_x_data /= 255
        testing_x_data /= 255
      
 
        print('printing training_x_data shapes',np.shape(training_x_data))
        print('printing testing_x_data shapes',np.shape(testing_x_data))
        print('printing Xdata_Testing shapes',np.shape(Xdata_Testing))
        
        print('printing training_y_data shapes',np.shape(training_y_data))
        print('printing testing_y_data shapes',np.shape(testing_y_data))
        print('printing ylables_Testing shapes',np.shape(ylables_Testing))
        
                      
        model = Sequential()
        
        model.add(Conv2D(8, (3, 3), input_shape=(30,30,3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(8, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(16,(3,3 )))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
                
        model.add(Flatten())
        
        # Fully connected layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.7))
        model.add(Dense(46))
        
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
        
        history = model.fit(training_x_data,training_y_data, epochs=4,batch_size = 32,validation_data=(testing_x_data,testing_y_data))
       
        score = model.evaluate(Xdata_Testing, ylables_Testing)
        print()
        print('Test loss: ', score[0])
        print('Test Accuracy', score[1])
        
        
        print(history.history.keys())

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        
        predictions = model.predict_classes(Xdata_Testing)
        
        predictions = list(predictions)
        actuals = list(ylables_Testing)
        
        sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
        sub.to_csv('./output_cnn.csv', index=False)



def Question_LDA_QDA_KKN_GAUSSIANWITHPCA():
    
    
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()
    

#citation
#https://www.geeksforgeeks.org/ml-principal-component-analysispca/                  
    pca = PCA(n_components = 30)
    Xdata = pca.fit_transform(Xdata)
    Xdata_Testing = pca.transform(Xdata_Testing)
    
    plt.figure(figsize=(8,6))
    plt.scatter(Xdata[:,0],Xdata[:,1],cmap='plasma')
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    plt.show()
        
    
    explained_variance = pca.explained_variance_ratio_
    
    
    shullfeDataResponse=createShuffleSplit(Xdata,ylables,5,0.2)
    computeAlgoMagic(shullfeDataResponse,Xdata,ylables,"LDA_QDA_KKN_GAUSSIANWITHPCA",Xdata_Testing,ylables_Testing)   
    
    
    return

def PlotDataVisulationzation():
    
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()
      
    num_stats = defaultdict(int)
    
    for num in ylables:
        num_stats[num] += 1
    
    x = sorted(num_stats)
    y = [num_stats[num] for num in x]
    
    plt.figure(figsize=(8,5))
    plt.bar(x, height=y)
    plt.xlabel("Image Content")
    plt.xticks(rotation='vertical')
    plt.ylabel("Frequency")
    plt.title("Distribution of Training Fruits Dataset Images")
    
#    
    num_stats_train = defaultdict(int)
    
    for num in ylables_Testing:
        num_stats_train[num] += 1
    
    x = sorted(num_stats_train)
    y = [num_stats_train[num] for num in x]
    
    plt.figure(figsize=(8,5))
    plt.bar(x, height=y)
    plt.xlabel("Image Content")
    plt.xticks(rotation='vertical')
    plt.ylabel("Frequency")
    plt.title("Distribution of Testing Fruits Dataset Images")
       
    return 


def Question_RANDOM_FOREST():
    
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()
    
    pca = PCA(n_components = 30)
    Xdata = pca.fit_transform(Xdata)
    Xdata_Testing = pca.transform(Xdata_Testing)
    
    classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
    classifier.fit(Xdata, ylables)

# Predicting the Test set results
    y_pred = classifier.predict(Xdata_Testing)

# Making the Confusion Matrix
    cm = confusion_matrix(ylables_Testing, y_pred)
    print(cm)
    score = accuracy_score(ylables_Testing, y_pred)
    print(score)
    
    return


def QuestionKmeansClusteringowithPCA():
    
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()
    
    #637 and 1901
    Xdata = pd.DataFrame(Xdata)
    Xdata_Testing = pd.DataFrame(Xdata_Testing)
    

    ylables = pd.DataFrame(ylables)
    ylables_Testing = pd.DataFrame(ylables_Testing)
    
#    PICKING 50 % OF THE ACTUAL DATSET TO PREDICT THE CATEGORY 
    
#    Xdata = Xdata.iloc[0:11423]
#    ylables = ylables.iloc[0:11423]
    
 #citation
#https://www.geeksforgeeks.org/ml-principal-component-analysispca/                 
    pca = PCA(n_components = 30)
    Xdata = pca.fit_transform(Xdata)
    Xdata_Testing = pca.transform(Xdata_Testing)
    explained_variance = pca.explained_variance_ratio_
    
    print(explained_variance)
        
    wcss = []
    for i in range(1, 130):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(Xdata)
        wcss.append(kmeans.inertia_)
        
        
    plt.plot(range(1, 130), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
        
  
   # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = 46, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit(Xdata)
        
    # Visualising the clusters
    
    plt.figure('K-means with 46 clusters')
    plt.scatter(Xdata[:, 0], Xdata[:, 1], c=kmeans.labels_)
    plt.show()
  
    return

def QuestionSVM():
    
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()


#citation
#https://www.geeksforgeeks.org/ml-principal-component-analysispca/    
    pca = PCA(n_components = 20)
    Xdata = pca.fit_transform(Xdata)
    Xdata_Testing = pca.transform(Xdata_Testing)
    
        
    svc_model = SVC()
    svc_model.fit(Xdata,ylables)
    
    predictions = svc_model.predict(Xdata_Testing)
    
    
    print(accuracy_score(ylables_Testing, predictions))
    
    print(confusion_matrix(ylables_Testing,predictions))
    
    print(classification_report(ylables_Testing,predictions))
    
    
    return

def SimpleLogisticRegression():
    
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()
    
    
    Xdata = pd.DataFrame(Xdata)
    Xdata_Testing = pd.DataFrame(Xdata_Testing)
    
    ylables = pd.DataFrame(ylables)
    ylables_Testing = pd.DataFrame(ylables_Testing)
 
    
    Xdata = Xdata.iloc[0:984]
    Xdata_Testing  = Xdata_Testing.iloc[0:328]
    
      
    ylables = ylables.iloc[0:984]
    ylables_Testing  = ylables_Testing.iloc[0:328]
    
    print(ylables)
    print(ylables_Testing)
 
    ylables_Testing = np.array(ylables_Testing)
    label_encoder = LabelEncoder()
    ylables_Testing = label_encoder.fit_transform(ylables_Testing)
        
    ylables = np.array(ylables)
    label_encoder = LabelEncoder()
    ylables = label_encoder.fit_transform(ylables)
       

    classifier = LogisticRegression(random_state = 0)
    classifier.fit(Xdata, ylables)
    
    # Predicting the Test set results
    y_pred = classifier.predict(Xdata_Testing)
    print(y_pred)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(ylables_Testing, y_pred)
    print(cm)
    
    score = accuracy_score(ylables_Testing, y_pred)
    print(score)
    
     
    return

def HeirerchiealClustering():
        
    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()
                  
    Xdata = pd.DataFrame(Xdata)
    Xdata_Testing = pd.DataFrame(Xdata_Testing)
         
    ylables = pd.DataFrame(ylables)
    ylables_Testing = pd.DataFrame(ylables_Testing)
            
    print('This  would take sometime printing dendograms applied PCA dendogram code is commented')
            

    pca = PCA(n_components = 2)
    Xdata = pca.fit_transform(Xdata)
    Xdata_Testing = pca.transform(Xdata_Testing)
    explained_variance = pca.explained_variance_ratio_
    
#    Xdata = Xdata.iloc[0:984]
#    Xdata_Testing  = Xdata_Testing.iloc[0:328]
#                          
#    ylables = ylables.iloc[0:984]
#    ylables_Testing  = ylables_Testing.iloc[0:328]
    

#    plt.figure(figsize=(300, 50))
#    dendrogram = sch.dendrogram(sch.linkage(Xdata, method = 'ward'))
#    plt.title('Dendrogram')
#    plt.xlabel('Fruits')
#    plt.xticks(rotation='vertical')
#    plt.ylabel('Euclidean distances')
#    color
#    plt.show()
#  

    hc = AgglomerativeClustering(n_clusters = 46, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(Xdata)
    
    plt.scatter(Xdata[:,0], Xdata[:,1],c=hc.labels_.astype(float))
    
    
    print(y_hc)

     
    return
    

    
def XGBoost():
    

    Xdata,ylables,Xdata_Testing,ylables_Testing = getData()
    
    
    ylables_Testing = np.array(ylables_Testing)
    label_encoder = LabelEncoder()
    ylables_Testing = label_encoder.fit_transform(ylables_Testing)
        
    ylables = np.array(ylables)
    label_encoder = LabelEncoder()
    ylables = label_encoder.fit_transform(ylables)
    
    classifier = XGBClassifier()
    classifier.fit(Xdata, ylables)
    

    y_pred = classifier.predict(Xdata_Testing)


    cm = confusion_matrix(Xdata_Testing, ylables_Testing)
    
    accuracies = cross_val_score(estimator = classifier, X = Xdata_Testing, y = ylables_Testing, cv = 10)
    accuracies.mean()
    accuracies.std()
    return 

def FacesDataset():
    
    
    Xdata = np.load('Faces_resized/resized_all.npy')
    ylables = np.load('Faces_resized/resized_labels.npy')

    Xdata = np.reshape(Xdata,(10770, 30000))
    

    stratifiedShuffleSplit = StratifiedKFold(n_splits=10, random_state=2)
    for train_index, val_index in stratifiedShuffleSplit.split(Xdata,ylables): 

        training_x_data, testing_x_data = Xdata[train_index], Xdata[val_index] 
        training_y_data, testing_y_data = ylables[train_index], ylables[val_index]
        
        
        training_y_data = np.array(training_y_data)
        label_encoder = LabelEncoder()
        training_y_data = label_encoder.fit_transform(training_y_data)
        
        testing_y_data = np.array(testing_y_data)
        label_encoder = LabelEncoder()
        testing_y_data = label_encoder.fit_transform(testing_y_data)
        
        
        train_X = training_x_data.reshape(training_x_data.shape[0],100, 100, 3)
        test_X = testing_x_data.reshape(testing_x_data.shape[0], 100, 100, 3)
            
        y_train = to_categorical(training_y_data)
        y_test = to_categorical(testing_y_data)
        
        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
            
        train_X /= 255
        test_X /= 255
        
        print('printing train_X shapes',np.shape(train_X))
        print('printing test_X shapes',np.shape(test_X))
        print('printing y_train shapes',np.shape(y_train))
        print('printing y_test shapes',np.shape(y_test))
        
        #createModelwithShuffleSplit(x_train,x_test,y_train,y_test)
       
        model = Sequential()
        
        model.add(Conv2D(16, (3, 3), input_shape=(100,100,3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        
        model.add(Conv2D(16, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
    
        
        model.add(Conv2D(32,(3,3 )))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        
    
        
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        
            
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        
        model.add(Flatten())
        
        # Fully connected layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.7))
        model.add(Dense(100))
        
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
        
        
        model.fit(train_X,y_train, epochs=1,batch_size = 32,
                            validation_data=(test_X,y_test))
        
    
        
        score = model.evaluate(test_X, y_test)
        print()
        print('Test loss: ', score[0])
        print('Test Accuracy', score[1])


      
    return
   
################## Solutions ########################


PlotDataVisulationzation()  

QuestionOneCNN('question1')  
        
#Question_LDA_QDA_KKN_GAUSSIAN()
#    
#QuestionKmeansClusteringowithPCA()
#    
#Question_LDA_QDA_KKN_GAUSSIANWITHPCA()
#
#Question_RANDOM_FOREST()
#    
#QuestionSVM()
#    
#SimpleLogisticRegression()
#
#HeirerchiealClustering()
#
#XGBoost()
#
#FacesDataset()









