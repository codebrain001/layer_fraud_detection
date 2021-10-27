# Training model using developed feature sets

# Importing libaries
from typing import Any
import pandas as pd # For data wrangling
from sklearn.preprocessing import PowerTransformer # For feature scaling
from sklearn.model_selection import train_test_split # For splitting data 
from imblearn.over_sampling import SMOTE # For oversampling
from sklearn.metrics import classification_report, roc_auc_score #Performance metrics
import xgboost as xgb # Machine learning model to be utilized
from layer import Featureset, Train


def train_model(train: Train, tf: Featureset("fraud_detection_features")) -> Any:

    """Model train function
    This function is a reserved function that will be called by Layer when we want this model to be trained along with the parameters.
    Args:
        train (layer.Train): Represents the current train of the model,
            passed by Layer when the training of the model starts.
        tf (layer.Featureset): Layer will return a Featureset object,
            an interface to access the features inside the
            `transaction_features`
    Returns:
        model: A trained model object
    """
    data_df = tf.to_pandas()
    X = data_df.drop(["INDEX", "flag"], axis=1)
    y = data_df["flag"]   

    # Split data and log parameters
    random_state = 45
    test_size = 0.2
    train.log_parameter("random_state", random_state)
    train.log_parameter("test_size", test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, random_state=random_state)

    # Normalize the training features
    norm = PowerTransformer()
    X_train_norm = norm.fit_transform(X_train)
    X_train_norm = pd.DataFrame(X_train_norm, columns=X_train.columns)

    #Oversampling with SMOTE
    oversample = SMOTE()
    X_tr_resample, y_tr_resample = oversample.fit_resample(X_train_norm, y_train)

    #For better naming processing, let's rename the training sets
    X_train = X_tr_resample
    y_train = y_tr_resample

    train.register_input(X_train)
    train.register_output(y_train)

    # We will use `XGBoost` for this task with fixed parameters
    n_estimators = 200
    subsample = 0.9
    learning_rate =0.5
    max_depth = 4
    colsample_bytree = 0.7

    # Train model
    param = {
        'n_estimators':n_estimators,
         'subsample':subsample, 
         'learning_rate':learning_rate,
         'max_depth': max_depth,
         'colsample_bytree':colsample_bytree
         }
    d_train = xgb.DMatrix(X_train, y_train)
    xgb_clf = xgb.train(param, d_train)

    train.log_parameter('n_estimators', n_estimators)
    train.log_parameter('subsample', subsample)
    train.log_parameter('learning_rate', learning_rate)
    train.log_parameter('max_depth', max_depth)
    train.log_parameter('colsample_bytree', colsample_bytree)

    # Transform test features
    X_test_norm = norm.transform(X_test)
    X_test = X_test_norm

    dtest = xgb.DMatrix(X_test)
    y_pred = xgb_clf.predict(dtest)

    # Track performance
    classification_report = classification_report(y_test, y_pred)
    roc_auc_score = roc_auc_score(y_test, y_pred)
    train.log_metric("classification_report", classification_report)
    train.log_metric("roc_auc_score", roc_auc_score)

    return xgb_clf





    
   
