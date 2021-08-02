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
    print('\033[1m  Type of Arms used by the deceases \033[0m')
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
    explode = (0.1, 0, 0, 0,0,0,0) 
    armsLables = 'Firearm', 'No', 'Knife', 'Other', 'Vehicle', 'Non-lethal firearm', 'Disputed'
    colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 7)
    plt.pie(armsData,explode=explode,shadow=True,labels=armsLables, autopct='%1.1f%%',colors=colors)
    print(' \033[1m  Most of the deceased have Fire arm with them. \033[1m ')

    plt.axis('equal')
    plt.title('Arms Distribution')
    plt.show()
    print('\n')

    gender_values = df["gender"].value_counts()
    gender_labels = 'Male', 'Female'
    explode = (0.1, 0) 
    colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 2)
    plt.pie(gender_values, explode=explode,shadow=True, labels=gender_labels, autopct='%1.1f%%',colors=colors)
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
    explode = (0.1, 0, 0, 0, 0) 
    colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = 5)
    race_labels = 'White', 'Black', 'Hispanic/Latino', 'Asian/Pacific Islander', 'Native American'
    plt.pie(race_values, explode=explode, shadow=True, labels=race_labels, autopct='%1.1f%%',colors=colors)
    plt.title('Breakdown by Race')
    plt.axis('equal')
    plt.show()
    print(' \033[1m  Most of the deceased are white. \033[1m ')
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
    plt.bar(x=state_labels, height=state_values,color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.title('Breakdown by State')
    plt.show()
    print(' \033[1m  Most of the deceased belong to CA. \033[1m ')
    print('\n')

    # city_values = df["city"].value_counts().head(5)
    # city_labels = 'Los Angeles', 'Houston', 'Phoenix', 'New York', 'Oklahoma City'
    # plt.bar(x=city_labels, height=city_values)
    # plt.xticks(rotation=45)
    # plt.title('Breakdown by City')
    # plt.show()
    # print('\n')

def df_correlation(df):
    print('\033[1m ************************ Correlation Between Attributes ************************ \033[0m')
    print('\n')
    
    incidents_per_city = df["city"].value_counts()
    avg_Pincome_perCity = df[['city', 'p_income']]
    avg_Pincome_perCity = avg_Pincome_perCity.groupby(
        [avg_Pincome_perCity["city"]]).mean()
    corr_city_avg_Pincome = pd.concat(
        [incidents_per_city, avg_Pincome_perCity], axis=1)
    #print(corr_city_avg_Pincome)
    correlation = corr_city_avg_Pincome['city'].corr(
        corr_city_avg_Pincome['p_income'])
    print('\n')
    print(
        'Number_of_incidents_in_a_city       Average_personal_income_in_city is : ' + str(
            round(correlation,2)))
    print('\n')

    avg_Hincome_perCity = df[['city', 'h_income']]
    avg_Hincome_perCity = avg_Hincome_perCity.groupby(
        [avg_Hincome_perCity["city"]]).mean()
    corr_city_avg_hincome = pd.concat(
        [incidents_per_city, avg_Hincome_perCity], axis=1)
    #print(corr_city_avg_hincome)
    correlation = corr_city_avg_hincome['city'].corr(
        corr_city_avg_hincome['h_income'])
    print(
        'Number_of_incidents_in_a_city       Average_HouseHold_income_in_city is : ' + str(
            round(correlation,2)))
    print('\n')

    avg_urate = df[['city', 'urate']]
    avg_urate = avg_urate.groupby(
        [avg_urate["city"]]).mean()
    corr_avg_urate_incidents = pd.concat(
        [incidents_per_city, avg_urate], axis=1)
    #print(corr_avg_urate_incidents)
    correlation = corr_avg_urate_incidents['city'].corr(
        corr_avg_urate_incidents['urate'])
    print(
        'Number_of_incidents_in_a_city       Average_Unemployment_rate is : ' + str(
            round(correlation,2)))
    print('\n')            

    avg_lrate = df[['city', 'college']]
    avg_lrate = avg_lrate.groupby(
        [avg_lrate["city"]]).mean()
    corr_avg_lrate_incidents = pd.concat(
        [incidents_per_city, avg_lrate], axis=1)
    #print(corr_avg_lrate_incidents)
    correlation = corr_avg_lrate_incidents['city'].corr(
        corr_avg_lrate_incidents['college'])
    print(
        'Number_of_incidents_in_a_city       Average_literacy_rate is : ' + str(
            round(correlation,2)))
    print('\n')
     
    avg_prate = df[['city', 'pov']]
    avg_prate = avg_prate.groupby(
        [avg_prate["city"]]).mean()
    corr_avg_prate_perCity = pd.concat(
        [incidents_per_city, avg_prate], axis=1)
    #print(corr_avg_prate_perCity)
    correlation = corr_avg_prate_perCity['city'].corr(
        corr_avg_prate_perCity['pov'])
    print(
        'Number_of_incidents_in_a_city       Average_poverty_rate in the city is : ' + str(
            round(correlation,2)))
    print('\n')

def decisionTree(data):
    print('\033[1m ************************ Decision Tree ************************ \033[0m')
    numeric_data = data.select_dtypes(include=np.number)
    X = numeric_data  
    y = data.iloc[:, 2] 
    # print(X.isnull().sum())
    selectBest = SelectKBest(score_func=chi2, k=5)
    fit = selectBest.fit(X, y)
    scores = pd.DataFrame(fit.scores_)
    columnss = pd.DataFrame(X.columns)
    f_scores = pd.concat([columnss, scores], axis=1)
    f_scores.columns = ['Factors', 'Score']
    dt_numeric_data = numeric_data[['h_income', 'county_income', 'p_income', 'pop', 'pov']]
    dt_data = data[['raceethnicity', 'armed', 'cause']]
    dt_data = pd.concat([dt_numeric_data, dt_data], axis=1)
    #print(dt_data)
    #print(dt_data['cause'].unique())
    x_train, x_test, y_train, y_test = dt_test_train_split(dt_data)
    dt_classification(x_train, x_test, y_train, y_test, 2)
    dt_classification(x_train, x_test, y_train, y_test, 3)
    dt_classification(x_train, x_test, y_train, y_test, 4)
    dt_classification(x_train, x_test, y_train, y_test, 5)
    dt_classification(x_train, x_test, y_train, y_test, 0)

def dt_test_train_split(df):
    df = df.apply(LabelEncoder().fit_transform)
    x = df.iloc[:, :7]
    y = df.iloc[:, 7]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def dt_classification(x_train, x_test, y_train, y_test, maxDepth):
    print("\033[1m Decision tree with depth \033[0m ", +maxDepth)
    if maxDepth>0:
        model = DecisionTreeClassifier(random_state=0, max_depth=maxDepth)
    else:
        model = DecisionTreeClassifier(random_state=0)
    model.fit(x_train, y_train)
    fn = ['h_income', 'county_income', 'p_income', 'pop', 'pov', 'raceethnicity', 'armed']
    cn = ['Gunshot', 'Death in custody', 'Taser', 'Struck by vehicle']
    fig = plt.figure(figsize=(25,20))
    tree.plot_tree(model, feature_names=fn, class_names=cn, filled=True)
    if maxDepth>0:
       plt.savefig('Decision_Tree_Depth-' + str(maxDepth) + '.png')
    train_accuracy = model.score(x_train, y_train)
    #print("Training Accuracy: ", +train_accuracy)
    y_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", +test_accuracy)
    print('\n')

def KmeansClustering_H_income(df):
   
    df = df[['h_income', 'cause']]
    df = df.apply(LabelEncoder().fit_transform)
    dist = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        dist.append(kmeanModel.inertia_)
    plt.figure(figsize=(9, 4))
    plt.plot(K, dist, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    #print('\033[1m ************************ KMeans with Household income and Cause ************************ \033[0m')
    plt.show()
    # The optimal k value is found out to be 3 based on elbow method.
    # Using k value as 3 while performing K-means clustering.
    kmeanModel = KMeans(n_clusters=3)
    kmeanModel.fit(df)
    print(df)
    df['k_means'] = kmeanModel.predict(df)
    print(df)
    fig, axes = plt.subplots(1, figsize=(12, 6))
    axes.scatter(df['h_income'], df['cause'], c=df['k_means'],
                 cmap=plt.cm.Set1)
    axes.set_title('K_Means', fontsize=18)
    plt.xlabel('Household Income')
    plt.ylabel('Cause ( 0.0 - Gunshot, 1.0 – Death in custody, 2.0 – Taser, 3.0 – Struck by vehicle )')
    plt.show()
    print('\n')


def KmeansClustering_P_income(df):
    print('\033[1m ************************ KMeans with Personal income and Cause ************************ \033[0m')
    df = df[['p_income', 'cause']]
    df = df.apply(LabelEncoder().fit_transform)
    dist = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        dist.append(kmeanModel.inertia_)
    plt.figure(figsize=(9, 4))
    plt.plot(K, dist, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()
    kmeanModel = KMeans(n_clusters=3)
    kmeanModel.fit(df)
    print(df)
    df['k_means'] = kmeanModel.predict(df)
    print(df)
    fig, axes = plt.subplots(1, figsize=(12, 6))
    axes.scatter(df['p_income'], df['cause'], c=df['k_means'],
                 cmap=plt.cm.Set1)
    axes.set_title('K_Means', fontsize=18)
    plt.xlabel('Personal Income')
    plt.ylabel('Cause ( 0.0 - Gunshot, 1.0 – Death in custody, 2.0 – Taser, 3.0 – Struck by vehicle )')
    plt.show()
    print('\n')


def hierarchialclustering(df):
    print('\033[1m ************************ Hierarchial clustering of cause and armed ************************ \033[0m')
    df = df[['armed', 'cause']]
    df = df.apply(LabelEncoder().fit_transform)
    # dendrogram to find number of clusters.
    dendrogram = sch.dendrogram(sch.linkage(df, method="ward"))
    plt.title('Dendrogram')
    plt.xlabel('Police Killings')
    plt.ylabel('Euclidean distances')
    plt.show()
    hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(df)
    plt.figure(figsize=(10, 7))
    plt.scatter(df['armed'], df['cause'], c=hc.labels_)
    plt.title('Hierarchical Agglomerative clustering')
    plt.xlabel(' Armed ( 0 - No, 1 - Firearm, 2 - Non-lethal firearm, 3 - Other, 4 - Knife, 5 - Vehicle, 6 - Disputed)')
    plt.ylabel(' Cause ( 0.0 - Gunshot, 1.0 – Death in custody, 2.0 – Taser, 3.0 – Struck by vehicle )')
    plt.show()

def hierarchialclustering_Hincome(data):
    print('\033[1m ************************ Hierarchial clustering of House hold income and cause ************************ \033[0m')
    data = data[['h_income', 'cause']]
    data = data.apply(LabelEncoder().fit_transform)
    # dendrogram to find number of clusters.
    dendrogram = sch.dendrogram(sch.linkage(data, method="ward"))
    plt.title('Dendrogram')
    plt.xlabel('House Hold Income')
    plt.ylabel('Euclidean distances')
    plt.show()
    hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(data)
    plt.figure(figsize=(10, 7))
    plt.scatter(data['h_income'], data['cause'], c=hc.labels_)
    plt.title('Hierarchical Agglomerative clustering')
    plt.xlabel(' Household income Income')
    plt.ylabel(' Cause ( 0.0 - Gunshot, 1.0 – Death in custody, 2.0 – Taser, 3.0 – Struck by vehicle )')
    plt.show()


def kNearestNeighbour(df):
    print('\033[1m ************************ Knn************************ \033[0m')
    temp = df[['age', 'p_income', 'h_income', 'pov', 'comp_income', 'cause']]
    #print(temp)
    temp = temp.apply(LabelEncoder().fit_transform)
    train = temp.iloc[:, :5]
    test = temp.iloc[:, 5]
    null_columns = train.columns[train.isnull().any()]
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.20, random_state=55, shuffle=True)
    print('\n k=3 \n')
    KNeighborsModel = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute')
    KNeighborsModel.fit(X_train, y_train)
    confusionMatrix = confusion_matrix(y_test, KNeighborsModel.predict(X_test))
    truePositive = confusionMatrix[0][0]
    trueNegative = confusionMatrix[1][1]
    falseNegative = confusionMatrix[1][0]
    falsePositive = confusionMatrix[0][1]

    #print("Knn Confusion Matrix")
    #print(confusionMatrix)
    print("Testing Accuracy = ", (truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive))

    print(classification_report(y_test, KNeighborsModel.predict(X_test)))
    print('\n')
    print("Accuracy Score is:", accuracy_score(y_test, KNeighborsModel.predict(X_test)))
    print('\n')
    print('\n k=7 \n')
    knc = KNeighborsClassifier(n_neighbors=7)
    knc.fit(X_train, y_train)
    title = "Knn : Confusion Matrix"
    disp = plot_confusion_matrix(knc, X_test, y_test, cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    plt.show()
    print('\n')

def pca(df, olddf):
    print('\n')
    null_columns = df.columns[df.isnull().any()]
    print(df[null_columns].isnull().sum())
    print("Are any value null", df.isnull().values.any())

    labelencoder = LabelEncoder()

    X = df
    Y = olddf["cause"]
    Y = labelencoder.fit_transform(olddf['cause'])
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    pca = decomposition.PCA(n_components=2)  
    X_new = pca.fit_transform(X)  
    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(X[:, 0], X[:, 1], c=Y)
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    axes[0].set_title('Before PCA')
    axes[1].scatter(X_new[:, 0], X_new[:, 1], c=Y)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('After PCA')
    plt.show()
    print('Variance_ratio')
    print(pca.explained_variance_ratio_)
    print('components')
    print(abs(pca.components_))
    print('\n')


def main():
    pd.set_option('display.width', 800)
    pd.set_option('display.max_columns', None)
    df = loadData('./sample_data/police_killings.csv')

    removeColumns(df)
    df = missingValues(df)
    df_distribution(df)
    AttributeValueDistribution(df)
    police_Shooting_Distribution(df)
    df_correlation(df)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    tempDf = df.select_dtypes(include=numerics)

    decisionTree(df)
    KmeansClustering_H_income(df)
    KmeansClustering_P_income(df)
    hierarchialclustering_Hincome(df)
    hierarchialclustering(df)
    pca(tempDf, df)
    kNearestNeighbour(df)


if __name__ == "__main__":
    main()
