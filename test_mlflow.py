import mlflow
from urllib.parse import urlparse
from datetime import datetime
from mlflow import keras as mlflow_keras

# Set your remote MLflow tracking URI (hosted on EC2)
mlflow.set_tracking_uri("http://ec2-52-66-232-50.ap-south-1.compute.amazonaws.com:5000/")  # Replace with actual IP
mlflow.set_registry_uri("http://ec2-52-66-232-50.ap-south-1.compute.amazonaws.com:5000/")

# Define experiment name
experiment_name = "MLflow Connection Test"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

# Check backend store type
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

with mlflow.start_run(run_name=f"test_run_{datetime.utcnow().isoformat()}"):
    # Log dummy parameters
    mlflow.log_param("test_param", 123)
    mlflow.log_param("model_type", "VGG16")

    # Log dummy metrics
    mlflow.log_metric("test_accuracy", 0.85)
    mlflow.log_metric("test_loss", 0.42)

    # Log a dummy artifact file
    with open("dummy.txt", "w") as f:
        f.write("This is a test artifact for MLflow S3 logging.")

    mlflow.log_artifact("dummy.txt")

    # Log model only if not using local file store
    if tracking_url_type_store != "file":
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
        model.compile(optimizer="adam", loss="mse")
        mlflow.keras.log_model(model, artifact_path="model", registered_model_name="DummyModel")
    else:
        print("Skipping model registration: local file store detected.")