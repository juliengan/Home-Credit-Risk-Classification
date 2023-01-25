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

def train_and_save_xgb():
    """Train the XGBBoost and save the model - Loads the pipeline, Retrieve the processed data       
    .. WARNING::  Be careful to not tie knots when moving the snake!       
    :param int x: The x position where move to.        
    :param int y: The y position where move to.        
    """
    source_path = Path(__file__).resolve()
    #mlflow.set_tracking_uri(tracking_uri)
    """mlflow.end_run()

    print(mlflow.get_tracking_uri())

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("score", clf.score(X_test, y_test))
        mlflow.sklearn.log_model(clf, "credit-risk-classifier")"""
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
    gs = GridSearchCV(pipeline.named_steps["clf"], {"max_depth": [1, 1.3, 1.5]}, n_jobs=-1, cv=5, scoring="accuracy")
    gs.fit(train, y_train)
    print("best params : ",gs.best_params_['max_depth'])
    pipeline.set_params(clf__max_depth=gs.best_params_['max_depth'])
    print('re training the gradient booster...')
    pipeline.named_steps['clf'].fit(train, y_train)
    pickle.dump(pipeline, open(f'{root_dir}/models/pipe.pkl','wb'))

if __name__ == '__main__':
    # get arguments
    n_estimators=int(sys.argv[1])
    learning_rate=int(sys.argv[2])
    max_depth=int(sys.argv[3])
    random_state = int(sys.argv[4])
    tracking_uri = sys.argv[5]


    train_and_save_xgb()

