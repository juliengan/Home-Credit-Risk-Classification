#%%
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
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
import pickle
from pathlib import Path


source_path = Path(__file__).resolve()
root_dir = source_path.parent.parent.parent    

def read_data(path):
    """
    Reads data and sets/sorts the index
    
    """
    data = pd.read_csv(path,
                            infer_datetime_format=True,
                            on_bad_lines='warn',
                            skip_blank_lines=True)[:10]
    try:
        df = data.sort_index()
        df = df.set_index("SK_ID_CURR")
    except:
        print("Unexpected error:", sys.exc_info()[0])
    print('\n', df.dtypes)
    return df

def nan(df):
    """
    Handles NaN values"""
    print("Process Nan...")
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values
    for col in numeric_cols:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))
        if (pct_missing < 4):                                             #* if NaN < 4% : replace by median
            med = df[col].median()
            df[col] = df[col].fillna(med)
    df_non_numeric = df.select_dtypes(exclude=[np.number])              #* Repeat process with non numerics variables
    non_numeric_cols = df_non_numeric.columns.values
    for col in non_numeric_cols:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))
        if pct_missing < 4:
            med = df[col].median()
            df[col] = df[col].fillna(med)
    print(df.shape)
    return df

def fix_typos(df):
    """Fixes the typos : replaces the commas / converts in MAJ
    :param DataFrame df : dataset
    """
    print("Fixing Typos...")
    obj = [col  for col, dt in df.dtypes.items() if dt == object]
    for col in obj:
        df[obj] = df[obj].str.replace(',', '.')
        df[obj] = df[obj].str.upper()
        df[obj] = df[obj].str.strip()
    print(df.shape)
    return df

def multiple_format(df, mult_var=None): 
    """One-hot encode our categorical features
       :param DataFrame df : dataset 
       :param list of strings mult_var : list of categorical variables
    """                                
    print("Encoding categorical varible(s)...")
    if mult_var is not None:
        df = pd.get_dummies(data=df, columns=mult_var)
    print(df.shape)
    return df

def normalization(df):
    """Normalizes the data to get same order of magnitude
    """
    scaler = MinMaxScaler()
    scaler.fit_transform(df)
    return df

def suppressOutliers(df):
    """Suppresses outliers of dataset
        :param DataFrame df : dataset
    """
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

def custom_silhouette(estimator, X):
      print("{}   -     ".format(round(silhouette_score(X, estimator.predict(X)), 4)), end = '')
      return np.mean(silhouette_score(X, estimator.predict(X)))

def custom_DBScrore(estimator, X):
      print(round(davies_bouldin_score(X, estimator.predict(X)), 4))
      return np.mean(davies_bouldin_score(X, estimator.predict(X)))
      
def data_prep(df, filename, mult_var=None):   
    """Processes the dataset without the use of the pipeline
    :param DataFrame df : dataset
    :param string filename : path of the dataset (local)
    :param mult_var list : list of categorical features 
    """ 
    df = df.drop_duplicates(keep='last')            #* Keep only most recent duplicatas
    df = pd.get_dummies(data=df, columns=mult_var)
    df = nan(df)   
    df.to_csv(f"{root_dir}/data/interim/" + filename)   #* Process empty values based on several conditions
    df = normalization(df) 
    df = df.convert_dtypes()                        #* Assign good type for the modelling phase
    df = df.select_dtypes(exclude=['object'])       #* Remove Object and String columns who are irrelevant
    return df


def extract_processed_data():
    """Extracts the raw data and apply the pipeline to the training and testing data : fits and transforms the datasets
    It saves the intermediate dataset (for visualisation) and final processed one.
    """
    source_path = Path(__file__).resolve()
    root_dir = source_path.parent.parent.parent
    path = f'{root_dir}/data/raw/application_train.csv'
    path_test = f'{root_dir}/data/raw/application_test.csv'
    train, test = read_data(path), read_data(path_test)
    X_train, X_test = train.iloc[:, 1:240], train.iloc[:, 1:240]
    cat_features = pickle.load(open(f"{root_dir}/data/features/cat_features.pkl",'rb'))
    num_features = pickle.load(open(f"{root_dir}/data/features/num_features.pkl",'rb'))
    pipeline = pickle.load(open(f"{root_dir}/models/pipe.pkl", 'rb'))
    ##### Extract preprocessed data right before normalizing to visualise later on
    data_cleaned = data_prep(train,mult_var=cat_features, filename="train_before_normalisation.csv")
    data_cleaned2 = data_prep(test,mult_var =cat_features, filename="test_before_normalisation.csv")
    print('preprocessing train...')
    x_train_ = pipeline.named_steps["preprocessing"].fit_transform(X_train)
    print('preprocessing test...')
    x_test_ = pipeline.named_steps["preprocessing"].fit_transform(X_test)
    x_train_ = pd.DataFrame(x_train_)
    x_test_ = pd.DataFrame(x_test_)
    x_train_.to_csv(f"{root_dir}/data/processed/application_train.csv")
    x_test_.to_csv(f"{root_dir}/data/processed/application_test.csv")
    print("data processed successfully and saved !")
extract_processed_data()
# %%
