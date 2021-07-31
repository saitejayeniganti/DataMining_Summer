import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import decomposition



def loadData(filename):
    return pd.read_csv(filename, encoding='ISO-8859-1')

def df_distribution(df):
    print('\n')
    print('\033[1m ************************ Data Distribution ************************ \033[0m')
    print('\n')
    print('Type of Arms')
    print(df["armed"].value_counts())
    print('\n')
    print('Type of Deaths')
    print(df["cause"].value_counts())
    print('\n')
    print('Sex of the Deceased')
    print(df["gender"].value_counts())
    print('\n')
    print('Race of the Deceased')
    print(df["raceethnicity"].value_counts())
    print('\n')


def main():
    pd.set_option('display.width', 800)
    pd.set_option('display.max_columns', None)
    data = loadData('./sample_data/police_killings.csv')

    removeColumns(data)
    #print(data.isnull().sum())
    data = missingValues(data)
    df_distribution(data)

if __name__ == "__main__":
    main()


