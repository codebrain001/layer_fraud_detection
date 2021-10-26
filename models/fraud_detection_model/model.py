# Training model using develop feature sets.

#Importing libaries
from typing import Any
from sklearn.preprocessing import PowerTransformer 
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
    y = train_df["flag"]    

    random_state = 45
    test_size = 0.25
    train.log_parameter("random_state", random_state)
    train.log_parameter("test_size", test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, random_state=random_state)

    # Normalize the training features
    norm = PowerTransformer()
    X_train = norm.fit_transform(X_train)

    train.register_input(X_train)
    train.register_output(y_train)

    max_depth = 3
    objective = 'binary:logitraw'
    train.log_parameter("max_depth", max_depth)
    train.log_parameter("objective", objective)

    # Train model
    param = {'max_depth': max_depth, 'objective': objective}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model_xg = xgb.train(param, dtrain)

    dtest = xgb.DMatrix(X_test)
    preds = model_xg.predict(dtest)

    # Since the data is highly skewed, we will use the area under the
    # precision-recall curve (AUPRC) rather than the conventional area under
    # the receiver operating characteristic (AUROC). This is because the AUPRC
    # is more sensitive to differences between algorithms and their parameter
    # settings rather than the AUROC (see Davis and Goadrich,
    # 2006: http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf)
    auprc = average_precision_score(y_test, preds)
    train.log_metric("auprc", auprc)

    # We return the model
    return model_xg

