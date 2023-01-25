#%%
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from pathlib import Path
import warnings
import logging
import sys
import mlflow
import pickle
from sklearn.model_selection import GridSearchCV


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def train_and_save_xgb(n_jobs,cv,scoring,tracking_uri):
    """Train the XGBBoost and save the model - Loads the pipeline, Retrieve the processed data       
    .. WARNING::  Be careful to not tie knots when moving the snake!       
    :param int x: The x position where move to.        
    :param int y: The y position where move to.        
    """
    source_path = Path(__file__).resolve()
    #mlflow.set_tracking_uri(tracking_uri)
 
    root_dir = source_path.parent.parent.parent
    pipeline = pickle.load(open(f'{root_dir}/models/pipe.pkl', 'rb'))
    train = pd.read_csv(f"{root_dir}/data/processed/application_train.csv")
    int_train = pd.read_csv(f"{root_dir}/data/interim/train_before_normalisation.csv")
    X_train = train#.iloc[:, 1:240]
    y_train = int_train.TARGET
    source_path = Path(__file__).resolve()
    root_dir = source_path.parent.parent.parent

    #X_train = train.drop('TARGET',axis=1)#pd.concat([train, test], ignore_index=True).drop("TARGET", axis=1)
    print("training the gradient booster...")
    pipeline.named_steps['clf'].fit(train, y_train)
    print("getting the best params...")
    gs = GridSearchCV(pipeline.named_steps["clf"], {"max_depth": [1, 3, 5]}, n_jobs=n_jobs, cv=cv, scoring=scoring)
    gs.fit(train, y_train)
    print("best params : ",gs.best_params_['max_depth'])
    pipeline.set_params(clf__max_depth=gs.best_params_['max_depth'])
    print('re training the gradient booster...')

    mlflow.end_run()

    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run():
        print('starting mlflow tracking...')
        pipeline.named_steps['clf'].fit(train, y_train)

        mlflow.log_metric("score", pipeline.named_steps["clf"].score(X_train, y_train))
        mlflow.sklearn.log_model(pipeline.named_steps["clf"], "credit-risk-classifier")

if __name__ == '__main__':
    # get arguments
    n_jobs=int(sys.argv[1])
    cv=int(sys.argv[2])
    scoring = sys.argv[3]
    tracking_uri = sys.argv[4]


    train_and_save_xgb(n_jobs,cv,scoring,tracking_uri)
    print('END !')

