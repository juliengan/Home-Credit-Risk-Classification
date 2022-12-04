from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from pathlib import Path


source_path = Path(__file__).resolve()
data_dir = source_path.parent.parent

train = pd.read_csv(f"{data_dir}/data/processed/application_train.csv").set_index('SK_ID_CURR')
test = pd.read_csv(f"{data_dir}/data/processed/application_train.csv").set_index('SK_ID_CURR')

X_train, X_test = train.iloc[:, 1:245], train.iloc[:, 1:245]
y_train, y_test = train.TARGET, train.TARGET
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
clf.get_params()


print(classification_report(y_test,clf.predict(X_test)))

clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
y_pred


pd.set_option('display.max_columns', None)

test