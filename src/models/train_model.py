from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from pathlib import Path
import warnings
import logging
import sys
import mlflow
import joblib

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# get arguments
n_estimators=int(sys.argv[1])
learning_rate=int(sys.argv[2])
max_depth=int(sys.argv[3])
random_state = int(sys.argv[4])
tracking_uri = sys.argv[5]


source_path = Path(__file__).resolve()
root_dir = source_path.parent.parent.parent


train = pd.read_csv(f"{root_dir}/data/processed/application_train.csv").set_index('SK_ID_CURR')
test = pd.read_csv(f"{root_dir}/data/processed/application_train.csv").set_index('SK_ID_CURR')

sample_size=10
X_train, X_test = train.iloc[:sample_size, 1:245], test.iloc[:sample_size, 1:245]
y_train, y_test = train.TARGET[:sample_size], test.TARGET[:sample_size]


#mlflow.set_tracking_uri(tracking_uri)
mlflow.end_run()

print(mlflow.get_tracking_uri())

with mlflow.start_run():
    clf = GradientBoostingClassifier(
            n_estimators=n_estimators, 
            learning_rate=learning_rate,
            max_depth=max_depth, 
            random_state=random_state
        ).fit(X_train, y_train)

    clf.get_params()

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    mlflow.log_metric("score", clf.score(X_test, y_test))

    mlflow.sklearn.log_model(clf, "credit-risk-classifier")
        
