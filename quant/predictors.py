from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from quant.helper_fcn import calculate_probabilities
from quant.types import IAi

if TYPE_CHECKING:
    import os


class AiRegresor(IAi):  # stara Michalovo B. verze
    """Class for training and predicting."""

    model: xgb.XGBRegressor | xgb.XGBClassifier

    def __init__(self):
        """Create a new Model from a XGBClassifier."""
        self.model = xgb.XGBRegressor(objective="reg:squarederror", max_depth=10)

    def fit(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Train AI model."""
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            training_dataframe.to_numpy(),
            outcomes.to_numpy(),
            test_size=0.01,
            random_state=2,
            shuffle=True,
        )
        self.model.fit(x_train, y_train)

    def get_probabilities(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win]."""
        predicted_score_differences = self.model.predict(dataframe)
        return calculate_probabilities(predicted_score_differences)

    def save_model(self, path: os.PathLike) -> None:
        """Save ML model."""
        self.model.save_model(path)


class DummyPredictor(IAi):
    """For testing, home always win."""

    def fit(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Do nothing."""

    def get_probabilities(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return probabilities such home always win."""
        probabilities = [(1, 0)] * dataframe.shape[0]
        return pd.DataFrame(probabilities, columns=pd.Index(["WinHome", "WinAway"]))


class SimpleRegressor(nn.Module):
    """A simple feedforward neural network for regression tasks."""

    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        Initialize the neural network layers.

        Args:
            input_size (int): Number of input features.
            hidden_size (int, optional): Number of neurons in the hidden layer. Default is 64.

        """  # noqa: E501
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (predicted values).

        """
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class AiTorch(IAi):
    """Class for training and predicting using PyTorch."""

    model: SimpleRegressor
    criterion: nn.MSELoss
    optimizer: optim.Adam
    scaler: MinMaxScaler

    def __init__(self, input_size: int):
        """Initialize a new PyTorch model."""
        self.model = SimpleRegressor(input_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = MinMaxScaler()

    def fit(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Train AI model using PyTorch with normalized data."""
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            training_dataframe.to_numpy().astype(np.float32),
            outcomes.to_numpy().astype(np.float32),
            test_size=0.01,
            random_state=2,
            shuffle=True,
        )

        x_train = self.scaler.fit_transform(x_train)
        x_val = self.scaler.transform(x_val)

        train_dataset = TensorDataset(
            torch.from_numpy(x_train), torch.from_numpy(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        for epoch in range(100):  # Number of training epochs  # noqa: B007
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(x_batch).squeeze()
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()

    def get_probabilities(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win] with normalized data."""  # noqa: E501
        with torch.no_grad():
            matice = dataframe.to_numpy().astype(np.float32)
            inputs = self.scaler.transform(matice)
            inputs = torch.from_numpy(inputs)
            predicted_score_differences = self.model(inputs).squeeze().numpy()
        return calculate_probabilities(predicted_score_differences)

    def save_model(self, path: os.PathLike) -> None:
        """Save ML model."""
        torch.save(self.model.state_dict(), path)


# TODO mozna pridat keras
