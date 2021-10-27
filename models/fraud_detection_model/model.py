# Importing libraries
from typing import Any
import pandas as pd # For data wrangling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # For splitting data 
from sklearn.metrics import roc_auc_score # Performance metrics
from sklearn.ensemble import RandomForestClassifier # Machine learning model to be utilized
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
    # Saving parameters of your training runs that will then be viewable in the Layer Model Catalog UI.
    train.log_parameter("random_state", random_state)
    train.log_parameter("test_size", test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, random_state=random_state)

    # Scaling the training features
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_train_sc = pd.DataFrame(X_train_sc, columns=X_train.columns)

    # For better naming processing, let's rename the training sets
    X_train = X_train_sc

    # Define the model signature, which can then be used for determining the data lineage of this model.
    train.register_input(X_train)
    train.register_output(y_train)
    
    # Instantiating machine learning model
    rf_clf = RandomForestClassifier()
    # Fitting the model to the data
    rf_clf.fit(X_train, y_train)

    # Transform test features
    X_test_sc = sc.transform(X_test)
    X_test = X_test_sc

    # Making predictions
    y_pred = rf_clf.predict(X_test)

    # Track performance
    score = roc_auc_score(y_test, y_pred)

    # Save metrics of your training runs that will then be viewable in the Layer Model Catalog UI
    train.log_metric("roc_auc_score", score)

    # Return the model
    return rf_clf