import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from asyncore import read

#%matplotlib inline
## Data Preparation
def read_data(path):  #? date_col is a list

    data = pd.read_csv(path,
                            infer_datetime_format=True,
                            on_bad_lines='warn',
                            skip_blank_lines=True)

    try:
        df = data.sort_index()
        df = df.set_index("SK_ID_CURR")
    except:
        print("Unexpected error:", sys.exc_info()[0])
    print('\n', df.dtypes)
    return df
def nan(df):
    print("Process Nan...")
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values
    for col in numeric_cols:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))
        if (pct_missing < 4):                                             #* if NaN < 4% : replace by median
            med = df[col].median()
            df[col] = df[col].fillna(med)
        """if pct_missing >= 20:                                           #* if NaN > 20% : drop features
            df = df.drop(columns=[col])
        if (pct_missing < 20) & (pct_missing >= 4) :                    #* if NaN < 20% & > 4% : drop lines
            df = df.dropna(subset=[col])"""

    df_non_numeric = df.select_dtypes(exclude=[np.number])              #* Repeat process with non numerics variables
    non_numeric_cols = df_non_numeric.columns.values
    for col in non_numeric_cols:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))
        if pct_missing < 4:
            med = df[col].median()
            df[col] = df[col].fillna(med)
        """if pct_missing >= 20:
            df = df.drop(columns=[col])
        if pct_missing < 20 :
            df = df.dropna(subset=[col])"""
    print(df.shape)
    return df

def fix_typos(df):
    print("Fixing Typos...")
    obj = [col  for col, dt in df.dtypes.items() if dt == object]
    for col in obj:
        df[obj] = df[obj].str.replace(',', '.')
        df[obj] = df[obj].str.upper()
        df[obj] = df[obj].str.strip()
    print(df.shape)
    return df

def multiple_format(df, mult_var=None):                                 #* mult_var is a list
    print("Encoding categorical varible(s)...")
    if mult_var is not None:
        df = pd.get_dummies(data=df, columns=mult_var)
    print(df.shape)
    return df

def normalization(df):
    scaler = MinMaxScaler()
    scaler.fit_transform(df)
    return df

def suppressOutliers(df):
    clf = IsolationForest(random_state=42)
    param_grid = {'n_estimators': list(range(100, 1000, 10)), 
                'contamination': [0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08], 
                'bootstrap': [True, False]}        

    grid_isol = RandomizedSearchCV(clf, 
                                    param_grid,
                                    scoring=custom_silhouette,              #? Davies Bouldin Score     or      Silhouette Score  
                                    refit=True,
                                    cv=3, 
                                    return_train_score=True)
    best_model = grid_isol.fit(df.values)
    custom_silhouette(best_model, df.values)
    custom_DBScrore(best_model, df.values)
    print('Optimum parameters', best_model.best_params_)
    y_pred = best_model.predict(df.values)
    train_clustered = df.assign(Cluster=y_pred)
    train_clustered = train_clustered.replace({-1: "Anomaly", 1: "Regular"})
    train_clustered["Cluster"].value_counts()
    # TO DO return value

def custom_silhouette(estimator, X):
      print("{}   -     ".format(round(silhouette_score(X, estimator.predict(X)), 4)), end = '')
      return np.mean(silhouette_score(X, estimator.predict(X)))

def custom_DBScrore(estimator, X):
      print(round(davies_bouldin_score(X, estimator.predict(X)), 4))
      return np.mean(davies_bouldin_score(X, estimator.predict(X)))

def data_prep(df, mult_var=None):
    df = df.drop_duplicates(keep='last')            #* Keep only most recent duplicatas
    #df = fix_typos(df)                              #* Set a good typos for categorical features
    df = pd.get_dummies(data=df, columns=mult_var)
    #df = multiple_format(df, mult_var=None)         #* Encode categorical variables
    df = nan(df)                                    #* Process empty values based on several conditions
    df = normalization(df) 
    #df = suppressOutliers(df)
    df = df.convert_dtypes()                        #* Assign good type for the modelling phase
    df = df.select_dtypes(exclude=['object'])       #* Remove Object and String columns who are irrelevant
    #df = df.convert_dtypes()                        #* Assign good type for the modelling phase
                                                    # TODO: Verify order of functions and add Outliers Removal
    #print('\n', df.dtypes)
    return df
path = r'..\data\raw\application_train.csv'
path_test = r'..\data\raw\application_test.csv'

train = read_data(path)
test = read_data(path_test)

mult_var = ["NAME_CONTRACT_TYPE","CODE_GENDER", "NAME_TYPE_SUITE","NAME_INCOME_TYPE",
                                "NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE", "OCCUPATION_TYPE", 
                                "WEEKDAY_APPR_PROCESS_START","ORGANIZATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
                                "WALLSMATERIAL_MODE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "EMERGENCYSTATE_MODE"]

data_cleaned = data_prep(train,mult_var)
data_cleaned2 = data_prep(test,mult_var)

data_cleaned.to_csv("../data/processed/application_train.csv")
data_cleaned2.to_csv("../data/processed/application_test.csv")