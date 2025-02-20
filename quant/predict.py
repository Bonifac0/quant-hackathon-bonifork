from __future__ import annotations

from typing import TYPE_CHECKING

# handles predicting results
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics, model_selection

if TYPE_CHECKING:
    import os


class Ai:
    """Class for training and predicting."""

    model: xgb.XGBRegressor | xgb.XGBClassifier

    def __init__(self):
        """Create a new Model from a XGBClassifier."""
        self.initialized = False

    def train(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Return trained model."""
        if not self.initialized:
            self.model = xgb.XGBClassifier()
            self.initialized = True

        self.model = self.model.fit(training_dataframe, outcomes)

    def train_reg(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Return trained model."""
        if not self.initialized:
            self.model = xgb.XGBRegressor(objective="reg:squarederror", max_depth=10)
            self.initialized = True
            print(training_dataframe.columns)

        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            training_dataframe.to_numpy(),
            outcomes.to_numpy(),
            test_size=0.01,
            random_state=2,
            shuffle=True,
        )
        print(x_train.shape)
        self.model.fit(x_train, y_train)
        print("MAE:", metrics.mean_absolute_error(y_val, self.model.predict(x_val)))
        print(self.model.feature_importances_)

    def get_probabilities(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win]."""
        return self.model.predict_proba(dataframe.to_numpy())

    def get_probabilities_reg(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win]."""
        predicted_score_differences = self.model.predict(dataframe)
        return self.calculate_probabilities(predicted_score_differences)

    def save_model(self, path: os.PathLike) -> None:
        """Save ML model."""
        self.model.save_model(path)

    def home_team_win_probability(self, score_difference: float) -> float:
        """Calculate the probability of home team winning based on score difference."""
        slope = 0.8  # range optimal 0.1 to 1. liked 0.3 and 0.5 (maybe 1)
        return 1 / (1 + np.exp(-slope * score_difference))

    def calculate_probabilities(self, score_differences: np.ndarray) -> pd.DataFrame:
        """Calculate the probabilities of teams winning based on score differences."""
        probabilities = []

        for score_difference in score_differences:
            home_prob = self.home_team_win_probability(score_difference)
            away_prob = 1 - home_prob
            probabilities.append((home_prob, away_prob))

        return pd.DataFrame(probabilities, columns=pd.Index(["WinHome", "WinAway"]))
