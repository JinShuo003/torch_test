import numpy as np

from model.linear_regression import LinearRegressionModel
from train_utils.get_train_data import x_train, y_train
import torch
import torch.nn as nn


def train_model():
    input_dim = 1
    output_dim = 1

    model = LinearRegressionModel(input_dim, output_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 1000
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("------------------------------------training begin--------------------------------------")
    for epoch in range(epochs):
        epoch += 1

        inputs = torch.from_numpy(x_train).to(device)
        labels = torch.from_numpy(y_train).to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        if epoch % 50 == 0:
            print("epoch: {}, loss: {}".format(epoch, loss.item()))

    torch.save(model.state_dict(), "parameters/model.pkl")




