from filelock import FileLock
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.datasets.mnist import load_data
from ray import tune

class MyModel(Model):
    def __init__(self, hiddens=128):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(hiddens, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

def train_mnist(config):
    with FileLock(os.path.expanduser("~/.tune.lock")):
        (x_train, y_train), (x_test, y_test) = load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    model = MyModel(hiddens=config["hiddens"])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)
    accuracy = model.evaluate(x_test, y_test, verbose=2)[1]

    tune.report(mean_accuracy=accuracy)

if __name__ == "__main__":
    analysis = tune.run(
        train_mnist,
        config={"hiddens": tune.choice([64, 128, 256])}
    )
    print("Best config: ", analysis.get_best_config(metric="mean_accuracy", mode="max"))
