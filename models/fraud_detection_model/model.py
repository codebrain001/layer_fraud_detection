# Training model using develop feature sets.

#Importing libaries
from typing import Any

from sklearn.preprocessing import PowerTransformer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
# 
from sklearn.ensemble import RandomForestClassifier
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

    # We will use `RandomForestClassifier` for this task with fixed parameters
    estimators = 100
    random_forest = RandomForestClassifier(n_estimators=estimators)

    # We can log parameters of this train. Later we can compare
    # parameters of different versions of this model in the Layer
    # Web interface
    train.log_parameter("n_estimators", estimators)

    # We fit our model with the train and the label data
    random_forest.fit(X_train, y_train)

    # Let's calculate the accuracy of our model
    y_pred = random_forest.predict(X_test)
    # Since the data is highly skewed, we will use the area under the
    # precision-recall curve (AUPRC) rather than the conventional area under
    # the receiver operating characteristic (AUROC). This is because the AUPRC
    # is more sensitive to differences between algorithms and their parameter
    # settings rather than the AUROC (see Davis and Goadrich,
    # 2006: http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf)
    auprc = average_precision_score(y_test, y_pred)
    train.log_metric("auprc", auprc)

    # We return the model
    return random_forest
