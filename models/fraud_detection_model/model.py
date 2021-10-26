# Training model using develop feature sets.

#Importing libaries
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import xgboost as xgb
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
    # We create the training and label data
    train_df = tf.to_pandas()
    X = train_df.drop(["INDEX", "flag"], axis=1)
    Y = train_df["flag"]    
