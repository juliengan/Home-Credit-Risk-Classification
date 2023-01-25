app-BD
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>



Report  :


Part 1: Building an ML Project for Home Credit Risk Classification

The ML model for home credit risk classification has been built using the provided dataset. Best practices for production ready code were followed, including:

GIT was used for team collaboration and code and model versioning
The ML project workflow was separated into different scripts for data preparation, training, prediction, visualize, pipeline.

Data preparation  : This is a script for data preparation, which includes functions for reading in data, handling missing values, fixing typos, encoding categorical variables, normalizing the data, and suppressing outliers. The script also imports various libraries such as pandas, numpy, and scikit-learn, and uses them to perform these tasks. The script includes a pipeline that is used to fit the data, and it also defines its own preprocessing functions such as normalize(), nan() and multiple_format() to further process the data before it is fit to the pipeline. These functions are used to normalize the data, handle missing values, and one-hot encode categorical variables. The script also uses various techniques to suppress outliers, such as Isolation Forest. The script also uses the library XGBoost for classification.

Training : This script is for training a model using the Gradient Boosting Classifier from the scikit-learn library. It imports various libraries such as pandas, scikit-learn, and XGBoost. The script includes a function train_and_save_xgb() which loads a pre-defined pipeline, reads in the training data, splits the data into features and labels, trains the gradient boosting classifier, uses GridSearchCV to find the best parameters for the model, and then retrains the model with the best parameters. After this, the pipeline is saved so it can be used for future predictions. The pipeline is defined in a separate script and it's loaded here and the classifier (XGBClassifier) inside the pipeline is trained on the input data. The pipeline also includes preprocessing steps like imputation, encoding and scaling.

Prediction : This script is for making predictions using a pre-trained model, which is loaded from a pickle file. It imports various libraries such as pandas and scikit-learn. The script loads the pre-trained pipeline, reads in the testing data, and uses the pipeline to make predictions on the test data. The script also calculates the accuracy score and classification report of the predictions. It then saves the predictions to a csv file.

Vizualisation : Heatmap of the correlations between various features in the training data. The heatmap is created using the seaborn library and is plotted using matplotlib. The features being plotted include various statistics (mean, mode, median) of different features in the data such as income, credit, age, and various other demographic data. Additionally, the heatmap also includes other features such as the number of social circles, phone change and credit bureau data. The correlation coefficient is represented by the color of the cell, with darker colors indicating a stronger correlation. This visualization can help identify which features are most strongly correlated with the target variable, as well as which features may be redundant or not useful for the model.
Visualize the feature importances and explanations of the XGBoost model using the eli5 library. The provided code loads the saved model and preprocessor, and then utilizes the eli5.show_weights() function to clearly display the relative importance of each feature in the model. Additionally, the eli5.explain_prediction_xgboost() function is used to provide an explanation for a prediction made by the model on a test dataset.
The code is loading the model and preprocessor. It is then using the eli5.show_weights() function to display the feature importances, and eli5.explain_prediction_xgboost() function to explain a prediction made by the model on a test dataset. The function "get_processed" is used to process the data and extract the categorical feature names. It then calls the eli5.show_weights function to display the feature importances, and returns the processed data, all features, and categorical feature names.
Displaying rows of the test dataset, and the corresponding label of the rows. Then it uses the eli5 library's show_prediction function to display the prediction made by the model on the specific rows. It also uses the convert_to_lime_format function to convert the dataframe to a format that can be used by the Lime library.
Creating a LimeTabularExplainer object, which is a tool for explaining the predictions made by a machine learning model. It is initializing the explainer with various parameters, such as the input data, feature names, and categorical names. The input data is passed in the form of a transformed dataframe using the convert_to_lime_format() function and the values of the dataframe is passed as the first parameter of the explainer. The mode is set to "classification" as the model is a classification model. The feature names are passed as a list of column names of the dataframe. The categorical_names and categorical_features are passed as the keys and values of the dictionary of categorical names. The discretize_continuous parameter is set to True, which indicates that continuous features should be discretized. Finally, the random_state is set to 42. This will ensure that the results are reproducible.
This code is visualizing the feature importances and explanations of an XGBoost model using the eli5 and shap libraries. It starts by loading the model and preprocessor, then it uses the eli5.show_weights() function to display the feature importances, and eli5.explain_prediction_xgboost() function to explain a prediction made by the model on a test dataset. After that, it uses the LIME library to provide a more local interpretation of the model by training a linear model around the prediction, then it uses the SHAP library to provide a more global interpretation of the model by calculating the contribution of each feature to the prediction for all instances. The code also includes some visualizations to help understand the results.

Pipeline : Creating a pipeline that includes preprocessing and an XGBClassifier model. The preprocessing step includes imputing missing values with SimpleImputer, scaling numerical features with StandardScaler and encoding categorical features with OneHotEncoder. The pipeline also includes the XGBClassifier model. The pipeline is then pickled to be exported for use in other applications. It is also using some features and columns from the dataset.
Defines a list of features that are considered numerical features and will be used later in the pipeline for preprocessing. These features are taken from a dataset that contains information about loan applications. These features include information such as the number of children, the applicant's income, credit amount, annuity amount, goods price, population relative to region, age, days employed, days registered, days ID was published, car age, mobile phone ownership, family member count, and various other demographic and financial information about the applicant. These numerical features will be used to train and predict a model.









A documentation library, such as Sphinx, was used

Pickle was used to save a trained machine learning model to a file, so that it can be loaded and used again later

Part 2: Integrating MLFlow Library into the Project

MLFlow was installed in the python environment and added to the library requirements
Parameters and metrics of the model were tracked and the results were displayed in the local MLFlow UI for multiple runs
The code was packaged in a reusable and reproducible model format with MLFlow projects

Part 3: Integrating SHAP Library into the Project

SHAP was installed in the python environment and added to the library requirements
SHAP was used to explain model predictions 
Visualizations were created to show explanations for specific points of the dataset, for all points of the dataset at once
