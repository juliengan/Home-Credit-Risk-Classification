#%%
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

source_path = Path(__file__).resolve()
root_dir = source_path.parent.parent.parent

pipe = pickle.load(open(f'{root_dir}/models/pipe.pkl', 'rb'))
train = pd.read_csv(f"{root_dir}/data/processed/application_train.csv")#.set_index('SK_ID_CURR')
X_test = pd.read_csv(f"{root_dir}/data/processed/application_test.csv")#.set_index('SK_ID_CURR')
int_train = pd.read_csv(f"{root_dir}/data/interim/train_before_normalisation.csv")
y = int_train.TARGET


y_pred = pipe.named_steps["clf"].predict(X_test)
test = pd.read_csv(f"{root_dir}/data/raw/application_test.csv")
test['prediction'] = pd.Series(y_pred)
test.to_csv(f"{root_dir}/data/output/predictions.csv")
# %%
