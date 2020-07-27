#import libraries
from joblib import dump,load #to save model
from sklearn.ensemble import RandomForestRegressor #for creating random forest regression
from sklearn.metrics import mean_squared_error #to evaluate model using mean square method
from sklearn.model_selection import cross_val_score #to perform cross validation(model evaluation technique)
from sklearn.tree import DecisionTreeRegressor # for creating decion tree regression
from sklearn.linear_model import LinearRegression#for creating linear regression model
from sklearn.impute import SimpleImputer#to fill missing values
from sklearn.pipeline import Pipeline#for sequentially applying a list of transformation
from sklearn.preprocessing import StandardScaler#for standardization
import pandas as pd #for data manipulation and analysis
from pandas.plotting import scatter_matrix #visualize trends in data
import matplotlib.pyplot as plt #for plotting
import numpy as np#collection of high-level mathematical functions
from sklearn.model_selection import StratifiedShuffleSplit #for splitting data into training set and testing set

def visualize_data_base(data): #function to generate histogram
    data.hist(bins=50,figsize=(20,15))#constructing histogram of data
    plt.savefig("visualizations/Data_histogram"+ ".png",bbox_inches="tight") #saving histogram
    """
    There are 14 histograms.Each histogram correspond to each attribute.X axis of histogram shows the attribute value.
    Y axis of histogram shows the number of occurance of each attribute value
    """

def visualize_correlations(data):#function to visualize correlation
    attributes=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']#all attributes except 'MEDV'
    for i in attributes:#loop
        attribute_set=[i,'MEDV']#attribute set considering in this iteration
        scatter_matrix(data[attribute_set],figsize=(12,8)) #constructing scatter matrix
        data.plot(kind="scatter",x=i,y="MEDV",alpha=0.8)#plotting each attributes against 'MEDV'
        plt.savefig(f"visualizations/correlations/ {i}_MEDV_CORRELATION"+ ".png",bbox_inches="tight") #Saving figure in correlation directory

def train_test_splitting(data,test_ratio,seed_value,splitting_attribute):#function to split data inti train and test set
    index=StratifiedShuffleSplit(n_splits=1,test_size=test_ratio,random_state=seed_value)#Provides train/test indices to split data in train/test sets
    for train_index,test_index in index.split(data,data[splitting_attribute]):#perform stratifiedshufflesplit based on splitting_attribute in data
        train_set=data.loc[train_index]#updating train_set
        test_set=data.loc[test_index]#updating test set
    return train_set,test_set #retuning test and train set

def data_pipeline(train_set):#function for perform pipelining
    my_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('std_scaler',StandardScaler()),])#creating pipeline
    new_train_set=my_pipeline.fit_transform(train_set)#transforming with pipeine and creates a numpy array
    return new_train_set#returning transformed data

def Linear_Regression_model(preprocesed_data,train_set_labels):#function to fit data to linear regression model
    model=LinearRegression()#selecting model
    model.fit(preprocesed_data,train_set_labels)#fitting data to the linear regression model
    return model #returning model

def Decision_Tree_model(preprocesed_data,train_set_labels):#function to fit data to decision tree model
    model=DecisionTreeRegressor()#selecting model
    model.fit(preprocesed_data,train_set_labels)#fitting data to the linear regression model
    return model #returning model

def Random_forest_model(preprocesed_data,train_set_labels):#calling function to generate random forest model)
    model=RandomForestRegressor()#selecting model
    model.fit(preprocesed_data,train_set_labels)#fitting data to the linear regression model
    return model #returning model    

def evaluation_RMSE(model,preprocesed_data,train_set_labels):#function to evaluate model using RMSE
    predicted_values=model.predict(preprocesed_data) #getting predicted values
    MSE_of_model=mean_squared_error(train_set_labels,predicted_values)#finding mean square
    RMSE_of_model=np.sqrt(MSE_of_model)#finding root mean square
    return RMSE_of_model#returning value

def evaluation_cross_validation(model,preprocesed_data,train_set_labels):#function to evaluate mlodel using cross validation
    scores=cross_val_score(model,preprocesed_data,train_set_labels,scoring="neg_mean_squared_error",cv=10) #getting scores
    rmse_scores=np.sqrt(-scores)#root mean square of cross validation score
    return rmse_scores#returning score

def training(train_set):#deals with training data set
    train_set_labels=train_set["MEDV"].copy()#separating class labels of all entries in training data set
    train_set_data=train_set.drop("MEDV",axis=1)#removing class label from training set
    preprocesed_data=data_pipeline(train_set_data) #calling function to perform transformations
    linear_model=Linear_Regression_model(preprocesed_data,train_set_labels)#calling function to prepare linear regression model
    Tree_model=Decision_Tree_model(preprocesed_data,train_set_labels)#calling function to generate decision tree model
    forest_model=Random_forest_model(preprocesed_data,train_set_labels)#calling function to generate random forest model
    Linear_Regression_RMSE_error=evaluation_RMSE(linear_model,preprocesed_data,train_set_labels)#calling function to evaluate linear regression model using MSE
    print(f"\nLinear regression error using Mean Square Method {Linear_Regression_RMSE_error}")
    Tree_Regression_RMSE_error=evaluation_RMSE(Tree_model,preprocesed_data,train_set_labels)#calling function to evaluate tree regression model using MSE
    print(f"Decision tree regression error using Mean square method {Tree_Regression_RMSE_error}")     
    Random_forest_RMSE_error=evaluation_RMSE(forest_model,preprocesed_data,train_set_labels)#calling function to evaluate random forest model using MSE
    print(f"Random forest regression error using Mean square method {Random_forest_RMSE_error}")    
    Linear_Regression_cross_validation_error=evaluation_cross_validation(linear_model,preprocesed_data,train_set_labels)#calling function to evaluate linear regression using cross validation
    print(f"\nLinear regression error using cross validation method {Linear_Regression_cross_validation_error.mean()}") 
    Tree_Regression_cross_validation_error=evaluation_cross_validation(Tree_model,preprocesed_data,train_set_labels)#calling function to evaluate tree regression using cross validation
    print(f"Decision tree regression error using cross validation method {Tree_Regression_cross_validation_error.mean()}") 
    Random_forest_cross_validation_error=evaluation_cross_validation(forest_model,preprocesed_data,train_set_labels)#calling function to evaluate random forest regression using cross validation
    print(f"Decision tree regression error using cross validation method {Random_forest_cross_validation_error.mean()}") 
    print(f"selected model is Random forest")
    return forest_model#returning best performing model

def testing(our_model,test_set):#function to predict test data
    new_test_set=test_set.drop("MEDV",axis=1) #removing actual price from test data
    new_test_set_labels=test_set["MEDV"].copy() #separating labels from test data
    prepared_test_data=data_pipeline(new_test_set)#passing through pipe line
    predictions=our_model.predict(prepared_test_data)#predicting output
    MSE=mean_squared_error(new_test_set_labels,predictions)#calculating MSE
    RMSE=np.sqrt(MSE)#Calculating RMSE
    return predictions,MSE,RMSE#returning values

def process_data(data,test_ratio,seed_value,splitting_attribute):#function
    visualize_data_base(data)#calling function to visualize dataset
    visualize_correlations(data)#visualizing correlations in data
    train_set,test_set=train_test_splitting(data,test_ratio,seed_value,splitting_attribute)#calling function
    print(f"\n\nnumber of rows in test set: {len(test_set)}\nnumber of rows in training set :{len(train_set)} ")  
    our_model=training(train_set) #getting final model
    dump(our_model,'housing_model.joblib') #saving model
    predictions,MSE,RMSE=testing(our_model,test_set)#getting test data details
    actual_labels=test_set["MEDV"].copy() #getting copy of labels of test set
    print(f"\n MSE is {MSE}")
    print(f"\n RMSE is {RMSE}")
    print(f"\norginal test data labels { actual_labels }")
    print(f"\npredicted values {predictions}")

if __name__ == '__main__':#execute below code only if the file was run directly, and not imported 
    data=pd.read_csv("data.csv")#reading data file
    test_ratio=0.2 #data splitting ratio
    seed_value=42#to get the same random number multiple times for splitting
    splitting_attribute='CHAS'
    process_data(data,test_ratio,seed_value,splitting_attribute) #calling function

