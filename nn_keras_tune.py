import os
from filelock import FileLock
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.config import RunConfig
from ray.air.integrations.keras import ReportCheckpointCallback
import ray

# Initialize Ray for Colab
ray.init(ignore_reinit_error=True)

# Training function
def train_mnist(config):
    batch_size = 128
    num_classes = 10
    epochs = 5  # Reduced epochs for Colab runtime

    with FileLock(os.path.expanduser("~/.data.lock")):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(config["hidden"], activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=config["lr"], momentum=config["momentum"]),  # Fixed argument
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[ReportCheckpointCallback(metrics={"mean_accuracy": "accuracy"})],
    )

# Tuning function
def tune_mnist():
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(train_mnist, resources={"cpu": 1, "gpu": 0}),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=5,  # Adjust for Colab runtime limits
        ),
        run_config=RunConfig(
            name="exp",
            stop={"mean_accuracy": 0.99},
        ),
        param_space={
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9),
            "hidden": tune.randint(64, 512),
        },
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

# Main execution
if __name__ == "__main__":
    print("Starting Ray Tuning...")
    tune_mnist()
    ray.shutdown()
