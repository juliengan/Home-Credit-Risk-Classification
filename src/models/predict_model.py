#%%
import pickle
import pandas as pd
from pathlib import Path

def predict():
    """Retrieves the trained XGBClassifier and processed data and predicts the label. Saves the predictions in a csv 
    """
    source_path = Path(__file__).resolve()
    root_dir = source_path.parent.parent.parent

    pipe = pickle.load(open(f'{root_dir}/models/pipe.pkl', 'rb'))
    train = pd.read_csv(f"{root_dir}/data/processed/application_train.csv")#.set_index('SK_ID_CURR')
    X_test = pd.read_csv(f"{root_dir}/data/processed/application_test.csv")#.set_index('SK_ID_CURR')
    int_train = pd.read_csv(f"{root_dir}/data/interim/train_before_normalisation.csv")
    y = int_train.TARGET


    print("start predicting...")
    y_pred = pipe.named_steps["clf"].predict(X_test)
    test = pd.read_csv(f"{root_dir}/data/raw/application_test.csv")
    test['prediction'] = pd.Series(y_pred)
    print("export output...")
    test.to_csv(f"{root_dir}/data/output/predictions.csv")
    print("END!")


if __name__ == "__main__":
    predict()