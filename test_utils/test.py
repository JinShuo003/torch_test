import torch

from model.linear_regression import LinearRegressionModel
from train_utils.get_train_data import x_train
import numpy as np


def test_model():
    print("------------------------------------testing begin--------------------------------------")

    input_dim = 1
    output_dim = 1

    model = LinearRegressionModel(input_dim, output_dim)
    model.load_state_dict(torch.load("parameters/model.pkl"))

    test_input = x_train
    test_input = np.array(test_input, dtype=np.float32).reshape(-1, 1)
    test_input = torch.tensor(test_input)
    test_output = model(test_input).data.numpy()
    print(test_output)

