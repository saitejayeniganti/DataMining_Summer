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

def AttributeValueDistribution(df):
    print('\033[1m ************************ Attribute Value Distribution ************************ \033[0m')
    print('\n')
    age = pd.to_numeric(df.age, errors='coerce')
    plt.plot(df["age"].value_counts(),color='red')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age value Distribution')
    plt.show()
    print('\n')

    urate = pd.to_numeric(df.urate, errors='coerce')
    df[["urate"]] = df[["urate"]].apply(pd.to_numeric)
    plt.plot(df["urate"].value_counts(),color='red')
    plt.title('Unemployment rate value Distribution')
    plt.show()
    print('\n')

    college = pd.to_numeric(df.college, errors='coerce')
    df[["college"]] = df[["college"]].apply(pd.to_numeric)
    plt.plot(df["college"].value_counts(),color='red')
    plt.title('Literacy rate value Distribution')
    plt.show()
    print('\n')


def missingValues(df):
    print('\033[1m ************************ Missing Values ************************ \033[0m')
    print('\n')
    age = pd.to_numeric(df.age, errors='coerce')
    index = age.isna()
    df["age"] = df["age"].replace("Unknown", '0')
    df[["age"]] = df[["age"]].apply(pd.to_numeric)
    medianAge = df['age'].median()
    print('Age has a skewed distribution : Replaced with Median')
    print('\n')
    df["age"] = df["age"].replace(0, round(medianAge))
    
    print('Armed, Cause , Race Ethnicity : Unknown values are removed')
    print('\n')
    df = df[df.raceethnicity != 'Unknown']
    df = df[df.armed != 'Unknown']
    df = df[df.cause != 'Unknown']

    urate = pd.to_numeric(df.urate, errors='coerce')
    index = urate.isna()
    df[["urate"]] = df[["urate"]].apply(pd.to_numeric)
    median_urate = df['urate'].median()
    df.fillna({'urate': median_urate}, inplace=True)
    print('Unemployment Rate : Replaced with Median')
    print('\n')

    college = pd.to_numeric(df.college, errors='coerce')
    index = college.isna()
    df[["college"]] = df[["college"]].apply(pd.to_numeric)
  
    medianCollege = df['college'].median()
    df.fillna({'college': medianCollege}, inplace=True)

    share_white = pd.to_numeric(df.share_white, errors='coerce')
    index = share_white.isna()
    df["share_white"] = df["share_white"].replace("-", '0')
    df["share_black"] = df["share_black"].replace("-", '0')
    df["share_hispanic"] = df["share_hispanic"].replace("-", '0')

    df["p_income"] = df["p_income"].replace("-", '0')
    df[["p_income"]] = df[["p_income"]].apply(pd.to_numeric)
    median_PersonalIncome = df['p_income'].median()
    df["p_income"] = df["p_income"].replace(0, round(median_PersonalIncome))

    median_HouseIncome = df['h_income'].median()
    df.fillna({'h_income': median_HouseIncome}, inplace=True)
    print('Household income : Replaced with Median')
    print('\n')

    median_CompIncome = df['comp_income'].median()
    df.fillna({'comp_income': median_CompIncome}, inplace=True)

    df["pov"] = df["pov"].replace("-", '0')
    df[["pov"]] = df[["pov"]].apply(pd.to_numeric)
    medianPov = df['pov'].median()
    df["pov"] = df["pov"].replace(0, round(medianPov))
    print('poverty rate : Replaced with Median')

    return df



def main():
    pd.set_option('display.width', 800)
    pd.set_option('display.max_columns', None)
    data = loadData('./sample_data/police_killings.csv')

    removeColumns(data)
    #print(data.isnull().sum())
    data = missingValues(data)
    df_distribution(data)
    AttributeValueDistribution(data)
  

if __name__ == "__main__":
    main()


