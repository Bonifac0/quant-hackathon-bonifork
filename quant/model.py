from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from quant.bet import Betting
from quant.data import Data
from quant.data_helper import TeamData
from quant.helper_fcn import match_to_opp
from quant.predictors import AiRegresor, AiTorch, DummyPredictor  # noqa: F401
from quant.ranking.elo import Elo, EloByLocation
from quant.types import IModel, Match, Opp, Summary


class Model(IModel):
    """Main class."""

    TRAIN_SIZE: int = 2000
    FEATURES: int = 51

    RANKING_COLUMNS: tuple[str, ...] = ("EloByLocation",)
    TRAINING_DATA_COLUMNS: tuple[str, ...] = (*RANKING_COLUMNS, *TeamData.MATCH_COLUMNS)

    def __init__(self) -> None:
        """Init classes."""
        self.elo = Elo()
        self.elo_by_location = EloByLocation()
        self.betting_bot = Betting()
        # self.predictor = AiTorch(self.FEATURES)
        # self.predictor = AiRegresor()
        self.predictor = DummyPredictor()
        self.data = Data()
        self.season_number: int = 0
        self.stop_loss_limit: int = 0
        self.old_data: pd.DataFrame = pd.DataFrame()
        self.last_retrain = 0

    def update_models(self, games: pd.DataFrame) -> None:
        """Update ranking models."""
        for match in (Match(*row) for row in games.itertuples()):
            self.elo.add_match(match)
            self.elo_by_location.add_match(match)
            self.data.add_match(match)

    def place_bets(
        self,
        summ: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run main function.

        Update ranking models if new data are given.
        Train predictor once a month.
        Place a bets.

        """
        games_increment: Match = inc[0]
        summary = Summary(*summ.iloc[0])

        if games_increment.shape[0] > 0:
            increment_season = int(games_increment.iloc[0]["Season"])
            if self.season_number != increment_season:
                # reset ranking models when new season
                self.elo.reset()
                self.elo_by_location.reset()

                self.season_number = increment_season

            self.update_models(games_increment)

            # train predictors once a month
            month = pd.to_datetime(summary.Date).month
            if self.last_retrain != month:
                self.old_data = pd.concat(
                    [self.old_data, games_increment],
                ).iloc[-self.TRAIN_SIZE :]

                self.train_predictor(self.old_data)

                self.last_retrain = month
                self.stop_loss_limit = summary.Bankroll

        # matches we want bet to
        # opps are dataframe, but they are similar to Opp
        active_matches = cast(pd.DataFrame, opps[opps["Date"] == summary.Date])

        # print()
        # print(f"active matches: {active_matches.shape[0]}")

        if active_matches.shape[0] == 0 or summary.Bankroll < (
            self.stop_loss_limit * 0.9
        ):  # return no bets
            return pd.DataFrame(
                data=0,
                index=np.arange(active_matches.shape[0]),
                columns=pd.Index(["BetH", "BetA"], dtype="str"),
            )

        # test_data = self._create_numpy_array(active_matches)
        data_to_predict = self._create_dataframe(active_matches)
        probabilities = self.predictor.get_probabilities(data_to_predict)
        bets = self.betting_bot.get_betting_strategy(
            probabilities, active_matches, summary
        )

        new_bets = pd.DataFrame(
            data=bets,
            columns=pd.Index(["BetH", "BetA"], dtype="str"),
            index=active_matches.index,
        )

        return new_bets.reindex(opps.index, fill_value=0)

    def _create_dataframe(self, active_matches: pd.DataFrame) -> pd.DataFrame:
        """Get matches to predict outcome for."""
        return cast(
            pd.DataFrame,
            active_matches.apply(
                lambda x: self._get_match_parameters(match_to_opp(Match(0, *x))),
                axis=1,
            ),
        )

    def _create_numpy_array(self, active_matches: Opp) -> np.ndarray:
        """Get matches to predict outcome for in NumPy array format."""
        return np.vstack(
            active_matches.apply(
                lambda x: self._get_match_parameters(match_to_opp(Match(0, *x))), axis=1
            ).values  # Ensure a NumPy-compatible format
        )

    def _get_match_parameters(self, opp: Opp) -> pd.Series:
        """Transform opp to usefull format for training and predicting."""
        elo_by_location_prediction = self.elo_by_location.predict(opp)

        if elo_by_location_prediction is None:
            elo_by_location_prediction = 0

        rankings = pd.Series(
            [elo_by_location_prediction],
            index=self.RANKING_COLUMNS,
        )

        data_parameters = self.data.get_match_parameters(opp)

        return pd.concat([rankings, data_parameters], axis=0)

    def train_predictor(self, dataframe: pd.DataFrame) -> None:
        """Train AI and add match to ranking."""
        training_data = []
        outcomes_list = []

        for match in (Match(*x) for x in dataframe.itertuples()):
            match_parameters = self._get_match_parameters(match_to_opp(match))

            if match_parameters is not None:
                training_data.append(match_parameters)
                outcomes_list.append(match.HSC - match.ASC)

        training_dataframe = pd.DataFrame(
            training_data, columns=pd.Index(self.TRAINING_DATA_COLUMNS)
        )

        outcomes = pd.Series(outcomes_list)

        self.predictor.fit(training_dataframe, outcomes)
