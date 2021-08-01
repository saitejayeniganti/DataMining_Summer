import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn import decomposition  
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random 
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans

def loadData(filename):
    return pd.read_csv(filename, encoding='ISO-8859-1')

def removeColumns(df):
    print('\033[1m ************************ Removing columns ************************ \033[0m')
    print('\n')
    print('Removed : name, county_id, county_fp, latitude, longitude, nat_bucket, tract_ce, streetaddress, day, geo_id, county_bucket, state_fp')
    print('\n')
    df.drop(
        ['name', 'county_id', 'county_fp', 'latitude', 'longitude', 'nat_bucket', 'tract_ce', 'streetaddress', 'day', 'geo_id', 'county_bucket',
          'state_fp'],
        axis=1, inplace=True)  

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


def df_distribution(df):
    print('\n')
    print('\033[1m ************************ Data Distribution ************************ \033[0m')
    print('\n')
    print('\033[1m  Type of Arms used by the deceased \033[0m')
    print(df["armed"].value_counts())
    print('\n')
    print('\033[1m  Type of Deaths \033[0m')
    print(df["cause"].value_counts())
    print('\n')
    print('\033[1m  Sex of the Deceased \033[0m')
    print(df["gender"].value_counts())
    print('\n')
    print('\033[1m  Race of the Deceased \033[0m')
    print(df["raceethnicity"].value_counts())
    print('\n')
    # print('\033[1m  Deceased distribution per state \033[0m')
    # print(df["state"].value_counts())
    # print('\n')
    print('\033[1m  Deceased distribution by Cause \033[0m')
    print(df["cause"].value_counts())
    print('\n')

def AttributeValueDistribution(df):
    print('\033[1m ************************ Attribute Value Distribution ************************ \033[0m')
    print('\n')
    age = pd.to_numeric(df.age, errors='coerce')
    plt.plot(df["age"].value_counts(),color='blue')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age value Distribution')
    plt.show()
    print('\n')

    urate = pd.to_numeric(df.urate, errors='coerce')
    df[["urate"]] = df[["urate"]].apply(pd.to_numeric)
    plt.plot(df["urate"].value_counts(),color='blue')
    plt.title('Unemployment rate value Distribution')
    plt.show()
    print('\n')

    college = pd.to_numeric(df.college, errors='coerce')
    df[["college"]] = df[["college"]].apply(pd.to_numeric)
    plt.plot(df["college"].value_counts(),color='blue')
    plt.title('Literacy rate value Distribution')
    plt.show()
    print('\n')

    pov = pd.to_numeric(df.pov, errors='coerce')
    df[["pov"]] = df[["pov"]].apply(pd.to_numeric)
    plt.plot(df["pov"].value_counts(),color='blue')
    plt.title('poverty rate value Distribution')
    plt.show()
    print('\n')

    hIncome = pd.to_numeric(df.pov, errors='coerce')
    df[["h_income"]] = df[["h_income"]].apply(pd.to_numeric)
    plt.plot(df["h_income"].value_counts(),color='blue')
    plt.title('House hold income value Distribution')
    plt.show()
    print('\n')

    pIncome = pd.to_numeric(df.pov, errors='coerce')
    df[["p_income"]] = df[["p_income"]].apply(pd.to_numeric)
    plt.plot(df["p_income"].value_counts(),color='blue')
    plt.title('Personal income value Distribution')
    plt.show()
    print('\n')

def police_Shooting_Distribution(df):
    print('\033[1m ************************ Killings Distribution ************************ \033[0m')
    print('\n')
    armsData = df["armed"].value_counts()
    armsLables = 'Firearm', 'No', 'Knife', 'Other', 'Vehicle', 'Non-lethal firearm', 'Disputed'
    colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 7)
    plt.pie(armsData, labels=armsLables, autopct='%1.1f%%',colors=colors)
    plt.axis('equal')
    plt.title('Arms Distribution')
    plt.show()
    print('\n')

    gender_values = df["gender"].value_counts()
    gender_labels = 'Male', 'Female'
    plt.pie(gender_values, labels=gender_labels, autopct='%1.1f%%')
    plt.title('Breakdown by Gender')
    plt.axis('equal')
    plt.show()
    if (gender_values.Male > gender_values.Female):
        print(' \033[1m  Most of the deceased belongs to Male group \033[1m ')
    else:
        print(' \033[1m  Most of the deceased belongs to Female group \033[1m ')

    print('\n')
    
    # ages = df["age"].value_counts(bins=10)
    # ages_labels = '(30.2, 37.3]', '(23.1, 30.2] ', '(37.3, 44.4]', '(15.928, 23.1]', '(44.4, 51.5]', '(51.5, 58.6]', '(58.6, 65.7]', '(65.7, 72.8] ', '(72.8, 79.9]', '(79.9, 87.0] '
    # plt.bar(x=ages_labels,
    #         height=ages)
    # plt.hist(ages_labels, rwidth=10)
    # plt.xticks(rotation=30)
    # plt.title('Breakdown by Ages')
    # plt.show()
    # print('\n')

    race_values = df["raceethnicity"].value_counts()
    race_labels = 'White', 'Black', 'Hispanic/Latino', 'Asian/Pacific Islander', 'Native American'
    plt.pie(race_values, labels=race_labels, autopct='%1.1f%%')
    plt.title('Breakdown by Race')
    plt.axis('equal')
    plt.show()
    print('\n')

    # month_values = df["month"].value_counts()
    # month_labels = 'March', 'April', 'February', 'January', 'May', 'June'
    # plt.bar(x=month_labels,
    #         height=month_values)
    # plt.title('Breakdown by Month')
    # plt.show()
    # print('\n')

    state_values = df["state"].value_counts().head(5)
    state_labels = 'CA', 'TX', 'FL', 'AZ', 'OK'
    plt.bar(x=state_labels, height=state_values)
    plt.title('Breakdown by State')
    plt.show()
    print('\n')

    # city_values = df["city"].value_counts().head(5)
    # city_labels = 'Los Angeles', 'Houston', 'Phoenix', 'New York', 'Oklahoma City'
    # plt.bar(x=city_labels, height=city_values)
    # plt.xticks(rotation=45)
    # plt.title('Breakdown by City')
    # plt.show()
    # print('\n')



def kNearestNeighbour(df):
    temp = df[['age', 'p_income', 'h_income', 'pov', 'comp_income', 'cause']]
    print(temp)
    temp = temp.apply(LabelEncoder().fit_transform)
    train = temp.iloc[:, :5]
    test = temp.iloc[:, 5]

    null_columns = train.columns[train.isnull().any()]
    print(train[null_columns].isnull().sum())
    print("Are any value null", train.isnull().values.any())
    print("y shape = ", train.shape)
    print(train)

    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.20, random_state=55, shuffle=True)

    KNeighborsModel = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute')

    KNeighborsModel.fit(X_train, y_train)

    confusionMatrix = confusion_matrix(y_test, KNeighborsModel.predict(X_test))

    truePositive = confusionMatrix[0][0]
    trueNegative = confusionMatrix[1][1]
    falseNegative = confusionMatrix[1][0]
    falsePositive = confusionMatrix[0][1]

    print("KNeighbours Algorithm confusion matrix")
    print(confusionMatrix)
    print("Testing Accuracy = ", (truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive))
    print()

    print(classification_report(y_test, KNeighborsModel.predict(X_test)))
    print("Accuracy Score is:", accuracy_score(y_test, KNeighborsModel.predict(X_test)))

    knc = KNeighborsClassifier(n_neighbors=7)
    knc.fit(X_train, y_train)
    title = "Knn : Confusion Matrix"
    disp = plot_confusion_matrix(knc, X_test, y_test, cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    plt.show()


def df_correlation(df):
    print('\033[1m ************************ Correlation Between Attributes ************************ \033[0m')
    print('\n')
    
    incidents_per_city = df["city"].value_counts()
    avg_Pincome_perCity = df[['city', 'p_income']]
    avg_Pincome_perCity = avg_Pincome_perCity.groupby(
        [avg_Pincome_perCity["city"]]).mean()
    corr_city_avg_Pincome = pd.concat(
        [incidents_per_city, avg_Pincome_perCity], axis=1)
    print(corr_city_avg_Pincome)
    correlation = corr_city_avg_Pincome['city'].corr(
        corr_city_avg_Pincome['p_income'])
    print('\n')
    print(
        'Correlation: Number_of_incidents_in_a_city vs Average_personal_income_in_city is : ' + str(
            correlation))
    print('\n')

    avg_Hincome_perCity = df[['city', 'h_income']]
    avg_Hincome_perCity = avg_Hincome_perCity.groupby(
        [avg_Hincome_perCity["city"]]).mean()
    corr_city_avg_hincome = pd.concat(
        [incidents_per_city, avg_Hincome_perCity], axis=1)
    print(corr_city_avg_hincome)
    correlation = corr_city_avg_hincome['city'].corr(
        corr_city_avg_hincome['h_income'])
    print(
        'Correlation: Number_of_incidents_in_a_city vs Average_HouseHold_income_in_city is : ' + str(
            correlation))
    print('\n')

    avg_urate = df[['city', 'urate']]
    avg_urate = avg_urate.groupby(
        [avg_urate["city"]]).mean()
    corr_avg_urate_incidents = pd.concat(
        [incidents_per_city, avg_urate], axis=1)
    print(corr_avg_urate_incidents)
    correlation = corr_avg_urate_incidents['city'].corr(
        corr_avg_urate_incidents['urate'])
    print(
        'Correlation: Number_of_incidents_in_a_city vs Average_Unemployment_rate is : ' + str(
            correlation))
    print('\n')            

    avg_lrate = df[['city', 'college']]
    avg_lrate = avg_lrate.groupby(
        [avg_lrate["city"]]).mean()
    corr_avg_lrate_incidents = pd.concat(
        [incidents_per_city, avg_lrate], axis=1)
    print(corr_avg_lrate_incidents)
    correlation = corr_avg_lrate_incidents['city'].corr(
        corr_avg_lrate_incidents['college'])
    print(
        'Correlation: Number_of_incidents_in_a_city vs Average_literacy_rate is : ' + str(
            correlation))
    print('\n')
     
    avg_prate = df[['city', 'pov']]
    avg_prate = avg_prate.groupby(
        [avg_prate["city"]]).mean()
    corr_avg_prate_perCity = pd.concat(
        [incidents_per_city, avg_prate], axis=1)
    print(corr_avg_prate_perCity)
    correlation = corr_avg_prate_perCity['city'].corr(
        corr_avg_prate_perCity['pov'])
    print(
        'Correlation: Number_of_incidents_in_a_city vs Average_poverty_rate in the city is : ' + str(
            correlation))
    print('\n')
   


def dt_test_train_split(df):
    # Apply label encoding
    df = df.apply(LabelEncoder().fit_transform)
    x = df.iloc[:, :7]
    y = df.iloc[:, 7]
    print("Shape of x: ", x.shape)
    print("Shape of y: ", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def main():
    pd.set_option('display.width', 800)
    pd.set_option('display.max_columns', None)
    data = loadData('./sample_data/police_killings.csv')

    removeColumns(data)
    #print(data.isnull().sum())
    data = missingValues(data)
    df_distribution(data)
    AttributeValueDistribution(data)
    police_Shooting_Distribution(data)
    df_correlation(data)
    


if __name__ == "__main__":
    main()
