#%%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost.sklearn import XGBClassifier
import pickle
import sys
import os
import pickle
from pathlib import Path

def read_data(path):
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

source_path = Path(__file__).resolve()
root_dir = source_path.parent.parent.parent
path = f"{root_dir}/data/raw/application_train.csv"
path_test = f"{root_dir}/data/raw/application_test.csv"
train = read_data(path)
y = train.TARGET
train, test = read_data(path), read_data(path_test)
X_train, X_test = train.iloc[:, 1:240], test.iloc[:, 1:240]
y_train, y_test = train.TARGET, train.TARGET

num_features = ["CNT_CHILDREN",	"AMT_INCOME_TOTAL",	"AMT_CREDIT",	"AMT_ANNUITY",	"AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE",	"DAYS_BIRTH",	"DAYS_EMPLOYED",
	"DAYS_REGISTRATION",	"DAYS_ID_PUBLISH",	"OWN_CAR_AGE",	"FLAG_MOBIL",	"FLAG_EMP_PHONE",	"FLAG_WORK_PHONE",	"FLAG_CONT_MOBILE",	"FLAG_PHONE",	"FLAG_EMAIL",
    "CNT_FAM_MEMBERS",	"REGION_RATING_CLIENT",	"REGION_RATING_CLIENT_W_CITY", "HOUR_APPR_PROCESS_START",	"REG_REGION_NOT_LIVE_REGION",	"REG_REGION_NOT_WORK_REGION",	
    "LIVE_REGION_NOT_WORK_REGION",	"REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY",	"LIVE_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_1",	"EXT_SOURCE_2",	"EXT_SOURCE_3",	"APARTMENTS_AVG",	"BASEMENTAREA_AVG",	"YEARS_BEGINEXPLUATATION_AVG",	"YEARS_BUILD_AVG",	
    "COMMONAREA_AVG",	"ELEVATORS_AVG",	"ENTRANCES_AVG",	"FLOORSMAX_AVG",	"FLOORSMIN_AVG",	"LANDAREA_AVG",	"LIVINGAPARTMENTS_AVG",	
    "LIVINGAREA_AVG",	"NONLIVINGAPARTMENTS_AVG",	"NONLIVINGAREA_AVG",	"APARTMENTS_MODE",	"BASEMENTAREA_MODE",	"YEARS_BEGINEXPLUATATION_MODE",	
    "YEARS_BUILD_MODE",	"COMMONAREA_MODE",	"ELEVATORS_MODE",	"ENTRANCES_MODE",	"FLOORSMAX_MODE",	"FLOORSMIN_MODE",	"LANDAREA_MODE",	
    "LIVINGAPARTMENTS_MODE",	"LIVINGAREA_MODE",	"NONLIVINGAPARTMENTS_MODE",	"NONLIVINGAREA_MODE",	"APARTMENTS_MEDI",	"BASEMENTAREA_MEDI",	
    "YEARS_BEGINEXPLUATATION_MEDI",	"YEARS_BUILD_MEDI",	"COMMONAREA_MEDI",
    "ELEVATORS_MEDI",	"ENTRANCES_MEDI",	"FLOORSMAX_MEDI",	"FLOORSMIN_MEDI",	"LANDAREA_MEDI",
    "LIVINGAPARTMENTS_MEDI",	"LIVINGAREA_MEDI",	"NONLIVINGAPARTMENTS_MEDI",	"NONLIVINGAREA_MEDI", "TOTALAREA_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE",	"DEF_30_CNT_SOCIAL_CIRCLE",	"OBS_60_CNT_SOCIAL_CIRCLE",	"DEF_60_CNT_SOCIAL_CIRCLE",	
    "DAYS_LAST_PHONE_CHANGE",	"FLAG_DOCUMENT_2",	"FLAG_DOCUMENT_3",	"FLAG_DOCUMENT_4",	"FLAG_DOCUMENT_5",	"FLAG_DOCUMENT_6",	
    "FLAG_DOCUMENT_7",	"FLAG_DOCUMENT_8",	"FLAG_DOCUMENT_9",	"FLAG_DOCUMENT_10",	"FLAG_DOCUMENT_11",	"FLAG_DOCUMENT_12",	
    "FLAG_DOCUMENT_13",	"FLAG_DOCUMENT_14",	"FLAG_DOCUMENT_15",	"FLAG_DOCUMENT_16",	"FLAG_DOCUMENT_17",	"FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19",	"FLAG_DOCUMENT_20",	"FLAG_DOCUMENT_21",	"AMT_REQ_CREDIT_BUREAU_HOUR",	"AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",	"AMT_REQ_CREDIT_BUREAU_MON",	"AMT_REQ_CREDIT_BUREAU_QRT",	"AMT_REQ_CREDIT_BUREAU_YEAR"	
	]

cat_features = ["NAME_CONTRACT_TYPE","CODE_GENDER", "NAME_TYPE_SUITE","NAME_INCOME_TYPE",
                    "NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE", "OCCUPATION_TYPE", 
                    "WEEKDAY_APPR_PROCESS_START","ORGANIZATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
                    "WALLSMATERIAL_MODE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "EMERGENCYSTATE_MODE"]



categorical_transformer = Pipeline(
    [
        ('imputer_cat', SimpleImputer(strategy = 'constant',
          fill_value = 'missing')),
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ]
)

numeric_transformer = Pipeline(
    steps=[
        ('imputer_num', SimpleImputer(strategy = 'median')),
        ('scaler', StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('categoricals', categorical_transformer, cat_features),
        ('numericals', numeric_transformer, num_features)
    ],
    remainder = 'drop'
)

pipeline = Pipeline(
        [
            ('preprocessing', preprocessor),
            ('clf', XGBClassifier(scale_pos_weight=(1 - y.mean()), n_jobs=-1))
        ]
)


print('fitting preprocessor')
# Some such as default would be binary features, but since
# they have a third class "unknown" we'll process them as non binary categorical
# Check if the directory exists
if not os.path.exists(f"{root_dir}/data/features"):
    os.makedirs(f"{root_dir}/data/features")
file_path = f"{root_dir}/data/features/num_features.pkl"
file_path2 = f"{root_dir}/data/features/cat_features.pkl"

if not os.path.isfile(file_path):
    with open(file_path, "wb") as f:
        pickle.dump(num_features, f)

if not os.path.isfile(file_path2):
    with open(file_path2, "wb") as f:
        pickle.dump(cat_features, f)

print('fitting one hot encoder...')
pipeline.named_steps['preprocessing'].transformers[0][1]\
   .named_steps['onehot']\
   .fit(X_train[cat_features],y_train)

print('fitting imputer for categocial features')
pipeline.named_steps['preprocessing'].transformers[0][1]\
    .named_steps['imputer_cat']\
    .fit(X_train[cat_features],y_train)


print('fitting imputer for numerical features')
pipeline.named_steps['preprocessing'].transformers[1][1]\
    .named_steps['imputer_num']\
    .fit(X_train[num_features],y_train)

print('fitting standard scaler...')
pipeline.named_steps['preprocessing'].transformers[1][1]\
   .named_steps['scaler']\
   .fit(X_train[num_features],y_train)

pickle.dump(pipeline, open(f"{root_dir}/models/pipe.pkl",'wb'))
print('model saved. OK')

# %%
