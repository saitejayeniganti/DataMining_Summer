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

def police_Shooting_Distribution(df):
    print('\033[1m ************************ Killings Distribution ************************ \033[0m')
    print('\n')
    armsData = df["armed"].value_counts()
    armsLables = 'Firearm', 'No', 'Knife', 'Other', 'Vehicle', 'Non-lethal firearm', 'Disputed'
    colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 7)
    plt.pie(armsData, labels=armsLables, autopct='%1.1f%%',colors=colors)
    plt.axis('equal')
    plt.title('Arms Distribution')
    print('\n')
    plt.show()
    print('\n')
    gender_values = df["gender"].value_counts()
    print('\n')
    if (gender_values.Male > gender_values.Female):
        print('Most of the deceased belongs to Male group')
    else:
        print('Most of the deceased belongs to Female group')
    gender_labels = 'Male', 'Female'
    plt.pie(gender_values, labels=gender_labels, autopct='%1.1f%%')
    plt.title('Breakdown by Gender')
    plt.axis('equal')
    plt.show()

    print('\n')
    ages = df["age"].value_counts(bins=10)
    ages_labels = '(30.2, 37.3]', '(23.1, 30.2] ', '(37.3, 44.4]', '(15.928, 23.1]', '(44.4, 51.5]', '(51.5, 58.6]', '(58.6, 65.7]', '(65.7, 72.8] ', '(72.8, 79.9]', '(79.9, 87.0] '
    plt.bar(x=ages_labels,
            height=ages)
    plt.hist(ages_labels, rwidth=10)
    plt.xticks(rotation=30)
    plt.title('Breakdown by Ages')
    plt.show()
    print('\n')

    race_values = df["raceethnicity"].value_counts()
    race_labels = 'White', 'Black', 'Hispanic/Latino', 'Asian/Pacific Islander', 'Native American'
    plt.pie(race_values, labels=race_labels, autopct='%1.1f%%')
    plt.title('Breakdown by Race')
    plt.axis('equal')
    plt.show()
    print('\n')

    month_values = df["month"].value_counts()
    month_labels = 'March', 'April', 'February', 'January', 'May', 'June'
    plt.bar(x=month_labels,
            height=month_values)
    plt.title('Breakdown by Month')
    plt.show()
    print('\n')

    state_values = df["state"].value_counts().head(5)
    state_labels = 'CA', 'TX', 'FL', 'AZ', 'OK'
    plt.bar(x=state_labels, height=state_values)
    plt.title('Breakdown by State')
    plt.show()
    print('\n')

    city_values = df["city"].value_counts().head(5)
    city_labels = 'Los Angeles', 'Houston', 'Phoenix', 'New York', 'Oklahoma City'
    plt.bar(x=city_labels, height=city_values)
    plt.xticks(rotation=45)
    plt.title('Breakdown by City')
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
    police_Shooting_Distribution(data)
   
   

if __name__ == "__main__":
    main()


