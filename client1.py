from typing import Dict, Tuple

import pandas as pd
import numpy as np
import flwr as fl
from flwr.common import Scalar, NDArrays

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
import tensorflow.keras.metrics as metrics

data = pd.read_pickle('data_part1.pkl')

features = np.asarray(data['image'].to_list())
target = np.asarray(data['dx_code'].to_list())

label_bin = LabelBinarizer()
label_bin.fit(range(max(target)+1))
encoded_target = label_bin.transform(target)

x_train, x_test, y_train, y_test = train_test_split(features, encoded_target, test_size=0.2)



model = tf.keras.applications.MobileNetV2((75, 100, 3), classes=7, weights=None)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])


class FlowerClient1(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters: NDArrays, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {'accuracy': accuracy}


fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient1())
#fl.client.start_client(server_address="[::]:8080", client=FlowerClient1.to_client())
