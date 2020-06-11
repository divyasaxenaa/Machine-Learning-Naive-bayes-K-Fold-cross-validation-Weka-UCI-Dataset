import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
import warnings

def readcsv(dataset_selected):
    url = "dataUpdated.csv"
    if dataset_selected == "car":
      names = ["Class","Buying","Maint","Doors","Persons","Lug_boot","Safety"]
    elif dataset_selected == "hayes":
        names = ["Name","Hobby", "Age", "Educational Level", "Marital Status", "Class"]
    elif dataset_selected == "iris":
        names = ["sepal length", "sepal width", "petal lengt", "petal width", "Class"]
    else:
        names = ["Class","Age", "Menopause", "Tumor-Size", "Inv-Nodes", "Node-Caps","Deg-Malig","Breast","Breast-Quad","Irradiat"]
    #data frame
    df = pd.read_csv(url, names=names,encoding='utf-8')
    print(df.describe())
    return df

def csvclear():
    lines = []
    with open('data.csv', 'r') as input:
        lines = input.readlines()

    conversion = '-" []"/$'
    newtext = ''
    outputLines = []
    for line in lines:
        temp = line[:]
        for c in conversion:
            temp = temp.replace(c, newtext)
        outputLines.append(temp)

    with open('dataUpdated.csv', 'w') as output:
        for line in outputLines:
            output.write(line + "\n")
    return 1

def preprocess(df):
    # Preprocess the data
    # ? =missing values in dataset; -9999 means ignore them
    df.replace('?',-9999,inplace=True)
    print(df.axes)
    #shape of dataset
    print(df.shape)
    #Dataset Visualization
    print(df.describe())
    #Plot histograms for each feature to understand distribution
    df.hist(figsize=(10,10))
    plt.show()
    return df

def scatter_mtrix(df):
    # Create scatter plot matrix
    #tells whether linear classifier will work good for the dataset or not
    #and to know realtionship between features
    scatter_matrix(df,figsize=(10,10))
    plt.show()
    return df

def main(dataset_selected):
    a = csvclear()
    df_csv = readcsv(dataset_selected)
    df = preprocess(df_csv)
    df_scatter =scatter_mtrix(df)
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
