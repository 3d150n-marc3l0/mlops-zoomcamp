import mlflow
import mlflow.sklearn
from os import path
import pickle
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("linear_regression_yellow_cab")
mlflow.sklearn.autolog()


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here

    dv = data[0]
    lr = data[1]
    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "model")
        
        # Serialize and log the DictVectorizer
        vectorizer_path = "dict_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(dv, f)
        
        # Log the DictVectorizer as an artifact
        mlflow.log_artifact(vectorizer_path, artifact_path="vectorizer")


        # Calculate and log the model size
        model_pickle_path = 'linear_regression_model.pkl'
        with open(model_pickle_path, 'wb') as f:
            pickle.dump(lr, f)

        # Log the model_size as an artifact
        model_size_bytes = os.path.getsize(model_pickle_path)
        print(model_size_bytes)
        mlflow.log_metric('model_size_bytes', model_size_bytes)
        
        print("Model and DictVectorizer logged to MLflow")

    os.remove(vectorizer_path)

