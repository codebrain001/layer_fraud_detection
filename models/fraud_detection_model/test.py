
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
    X = train_df.drop(["Index", "flag"], axis=1)
    Y = train_df["flag"]

    random_state = 45
    test_size = 0.25
    train.log_parameter("random_state", random_state)
    train.log_parameter("test_size", test_size)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_size,
                                                    random_state=random_state)

    # Here we register input & output of the train. Layer will use
    # this registers to extract the signature of the model and calculate
    # the drift
    train.register_input(trainX)
    train.register_output(trainY)

    max_depth = 3
    objective = 'binary:logitraw'
    train.log_parameter("max_depth", max_depth)
    train.log_parameter("objective", objective)

    # Train model
    param = {'max_depth': max_depth, 'objective': objective}
    dtrain = xgb.DMatrix(trainX, label=trainY)
    model_xg = xgb.train(param, dtrain)

    dtest = xgb.DMatrix(testX)
    preds = model_xg.predict(dtest)

    # Since the data is highly skewed, we will use the area under the
    # precision-recall curve (AUPRC) rather than the conventional area under
    # the receiver operating characteristic (AUROC). This is because the AUPRC
    # is more sensitive to differences between algorithms and their parameter
    # settings rather than the AUROC (see Davis and Goadrich,
    # 2006: http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf)
    auprc = average_precision_score(testY, preds)
    train.log_metric("auprc", auprc)

    # We return the model
    return model_xg