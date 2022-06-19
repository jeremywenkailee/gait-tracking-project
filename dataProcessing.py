from contextlib import nullcontext
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
import glob
import os

import warnings
warnings.filterwarnings("ignore")

def stringFilter(stringArg):
    stringPos = stringArg.find('_')
    stringArg = stringArg[:stringPos]
    return stringArg

#Reading the Raw Data
def readData(inputFolder):
    walkData = 0
    totalData = 0
    walkDataArray = []
    fileArray = glob.glob(os.path.join(inputFolder, '*.csv'))

    # print(os.path.join(inputFolder,"data_2.csv"))
    # print(fileArray)
    
    #Iterate through each file, perform transformation/normalization/filtering, push to array for concat.
    for x in fileArray:
        originalData = pd.read_csv(x, names=["time", "ax (m/s^2)", "ay (m/s^2)", "az (m/s^2)", "aT (m/s^2)"], header=0)
        nameClassifier = os.path.basename(x)
        originalData['filename'] = stringFilter(nameClassifier)
        originalData = originalData[['filename','time',"ax (m/s^2)", "ay (m/s^2)", "az (m/s^2)", "aT (m/s^2)"]]
        
        #processing
        walkData = filterAnalysis(originalData)
        walkData = normalize(walkData)
        walkData = stepAnalysis(walkData)
        # walkData['timeStamp'] =  walkData['timeStamp'].shift(1)
        walkData = trimColumns(walkData)
        # print(walkData)
        walkDataArray.append(walkData)

    totalData = pd.concat(walkDataArray)
    # totalData = walkDataArray
    return totalData

#Printing the Raw Data
def plotData(dataF, filterData = None):
    plt.figure(figsize=(80, 30))  # change the size to something sensible
    plt.title('Title')
    plt.xlabel('Time (S)')
    plt.ylabel("Linear Acceleration (m/s^2)")
    if (filterData is not None):
        plt.plot(dataF['time'].values, filterData, 'g')
    else:
        plt.plot(dataF['time'],dataF['normalized_aT'],'y.')
        plt.plot(dataF['time'],dataF['stepCount'],'g.')
    plt.show()

#normalize the data
def normalize(dataF):
    # print("Normalize")
    row_size, column_size = dataF.shape
    size = int(row_size * 0.05)
    dataF.drop(dataF.head(size).index, inplace=True)  # drop first n rows
    dataF.drop(dataF.tail(size).index, inplace=True)
    dataF['normalized_aT'] = (dataF['aT Filtered'] - dataF['aT Filtered'].min()) / (dataF['aT Filtered'].max() - dataF['aT Filtered'].min()) - (dataF['aT Filtered'].mean() - dataF['aT Filtered'].min()) / (dataF['aT Filtered'].max() - dataF['aT Filtered'].min())
    return dataF

#lambda func
def stepTally(x,steps,prev):
    timeStamp = False
    if (x > 0 and prev < 0) or (x < 0 and prev > 0):
        steps += 1
        timeStamp = True
    prev = x
    return steps, prev, timeStamp

#stepAnalysis
def stepAnalysis(dataF):
    # print("stepAnalysis")
    #TODO: Brad - Step analysis in a new column - return a new dataFrame
    step=0
    stepArray = []
    prev = 0
    timeStampArray = []

    for index, row in dataF.iterrows():
        step, prev, timeStamp = stepTally(row['normalized_aT'],step,prev)
        if(timeStamp):
            timeStampArray.append(row['time'])
        else:
            timeStampArray.append(0)
        stepArray.append(step)
    
    dataF['stepCount'] = stepArray
    dataF['timeStamp'] = timeStampArray

    #Trimming only the Step Entries with Timestamp and Acceleration at each time
    dataF = dataF[dataF['timeStamp'] != 0]

    #time per step
    dataF['timeStamp'] =  dataF['timeStamp'].shift(1)
    dataF['timeStamp'].iloc[0] = dataF['timeStamp'].iloc[1]
    dataF['stepTime'] = dataF['time'] - dataF['timeStamp']
    return dataF

def filterAnalysis(dataF):
    # TODO: Jerrick - Filtering data to remove noise - return a new dataFrame
    # print("filterAnalysis")
    # bad filters
    # b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
    # b, a = signal.butter(3, 0.2, btype='lowpass', analog=False)
    # b, a = signal.butter(3, 0.05, btype='lowpass', analog=False)
    # b, a = signal.butter(3, 0.02, btype='lowpass', analog=False)
    # b, a = signal.butter(3, 0.005, btype='lowpass', analog=False)

    # good filter
    b, a = signal.butter(3, 0.01, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, dataF["aT (m/s^2)"])
    dataF["aT Filtered"] = low_passed
    return dataF

def trimColumns(dataF):
    dataF = dataF.drop(columns = ["aT Filtered","timeStamp"])
    return dataF

def processColumns(dataF):

    return dataF

def modelAnalysis(dataF):
    # print("modelAnalysis")
    #TODO: Jeremy - Insert dataframe into a pipeline to produce an ML model
    X = dataF.drop(columns = ["time", "filename", "stepCount" ])
    y = dataF['filename']

    #split the data into valid and training data
    X_train, X_valid, y_train, y_valid = train_test_split(X,y)
    print("X_train structure:")
    print(X_train)
    print("y_train strucutre:")
    print(y_train)
    print(" ")
    #gaussian
    bayes_model = GaussianNB()
    #neighbours
    knn_model = KNeighborsClassifier(n_neighbors=4)
    knn_model_moreNeighbours = KNeighborsClassifier(n_neighbors=7)
    knn_model_mostNeighbours = KNeighborsClassifier(n_neighbors=13)

    #forest
    rf_model = RandomForestClassifier(30, max_depth = 4)
    rf_model_more_trees = RandomForestClassifier(50, max_depth = 4)
    rf_model_more_depth = RandomForestClassifier(30, max_depth = 8)
    rf_model_more_trees_more_depth = RandomForestClassifier(50,max_depth = 8)

    scores = []
    modelNames = ["Bayes Model", "KNN Model", "KNN with more Neighbours", "KNN with most Neighbours", "RF Model", "RF with more trees", "RF with more depth", "RF with more Trees and Depth"]
    models = [bayes_model,knn_model,knn_model_moreNeighbours,knn_model_mostNeighbours,rf_model,rf_model_more_trees,rf_model_more_depth,rf_model_more_trees_more_depth]
    for i, m in enumerate(models):
        m.fit(X_train,y_train)
        print(modelNames[i] + ":")
        print(m.score(X_valid,y_valid))
        print(" ")
        scores.append(m.score(X_valid,y_valid))
    print("MODEL WITH THE HIGHEST SCORE TO BE USED:" + modelNames[scores.index(max(scores))])
    
    return models[scores.index(max(scores))]




def main(trainFolder,predictFolder,output):
    print("Reading and Processing Data:")
    original_data = readData(trainFolder)
    print(original_data)
    # original_data.to_csv(output)

    print(" ")

    print("Modelling Data:")
    model = modelAnalysis(original_data)

    print(" ")
    
    print("Testing predictions")
    predict_data = readData(predictFolder)
    actual_name = predict_data['filename']
    # print(actual_name)
    predict_data = predict_data.drop(columns = ["time", "filename", "stepCount" ])
    predicty = model.predict(predict_data)

    dfpredicty = pd.DataFrame(predicty)
    dfpredicty = dfpredicty.reset_index(drop = True)
    actual_name = actual_name.reset_index(drop = True)
    dfpredicty = dfpredicty.join(actual_name)
    dfpredicty = dfpredicty.rename(columns={0: "predicted_values","filename":"actual_values"})
    # print(dfpredicty)
    dfpredicty['is_correct_prediction'] = dfpredicty['predicted_values'] == dfpredicty['actual_values']
    print(dfpredicty)
    print(" ")

    # print(dfpredicty['is_correct_prediction'].value_counts())
    amount_correct = int(dfpredicty['is_correct_prediction'].values.sum())
    amount_total = len(dfpredicty.axes[0])
    # print(amount_correct,amount_total,amount_correct/amount_total)
    # print(type(amount_correct),type(amount_total),type(amount_correct/amount_total))
    print("TOTAL CORRECT: " + str(amount_correct) + " OUT OF " + str(amount_total) + " : " + str(amount_correct/amount_total) + "% CORRECT")

    dfpredicty.to_csv(output, index = True, header=True)

    


if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])
