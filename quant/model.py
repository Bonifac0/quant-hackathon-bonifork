from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from quant.bet import Betting
from quant.data import Data
from quant.data_helper import TeamData
from quant.helper_fcn import match_to_opp
from quant.predictors import AiRegresor, AiTorch
from quant.ranking.elo import Elo, EloByLocation
from quant.types import IModel, Match, Opp, Summary


class Model(IModel):
    """Main class."""

    TRAIN_SIZE: int = 2000
    FIRST_TRAIN_MOD: int = 4
    RANKING_COLUMNS: tuple[str, ...] = (
        "HomeElo",
        "AwayElo",
        "EloByLocation",
    )
    MATCH_PARAMETERS = len(TeamData.COLUMNS) + len(RANKING_COLUMNS)
    TRAINING_DATA_COLUMNS: tuple[str, ...] = (*RANKING_COLUMNS, *TeamData.MATCH_COLUMNS)

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.elo = Elo()
        self.elo_by_location = EloByLocation()
        self.betting_bot = Betting()
        # self.ai = AiTorch(53)  # num of features (dimensions)
        self.ai = AiRegresor()
        self.trained = False
        self.data = Data()
        self.season_number: int = 0
        self.budget: int = 0
        self.old_matches: pd.DataFrame = pd.DataFrame()
        self.old_outcomes: pd.Series = pd.Series()
        self.last_retrain = 0

    def update_models(self, games_increment: pd.DataFrame) -> None:
        """Update models."""
        for match in (Match(*row) for row in games_increment.itertuples()):
            self.elo.add_match(match)
            self.elo_by_location.add_match(match)
            self.data.add_match(match)

    def place_bets(
        self,
        summ: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run main function."""
        games_increment: Match = inc[0]
        summary = Summary(*summ.iloc[0])

        if not self.trained:
            train_size = self.TRAIN_SIZE * self.FIRST_TRAIN_MOD
            print(
                f"Initial training on {games_increment[-train_size :].shape[0]}"
                f" matches with bankroll {round(summary.Bankroll, 2)}"
            )
            self.train_ai_predictor(cast(pd.DataFrame, games_increment[-train_size:]))

        elif games_increment.shape[0] > 0:
            increment_season = int(games_increment.iloc[0]["Season"])
            if self.season_number != increment_season:
                self.elo.reset()
                self.elo_by_location.reset()

                self.season_number = increment_season

            self.old_matches = pd.concat(
                [
                    self.old_matches.iloc[-self.TRAIN_SIZE :],
                    self._create_dataframe(games_increment),
                ],
            )

            self.old_outcomes = cast(
                pd.Series,
                pd.concat(
                    [
                        self.old_outcomes.iloc[-self.TRAIN_SIZE :],
                        games_increment.HSC - games_increment.ASC,
                    ],
                ),
            )

            month = pd.to_datetime(summary.Date).month
            if self.last_retrain != month:

                print(
                    f"{summary.Date.date()}: retraining on {self.old_matches.shape[0]}"
                    f" matches with bankroll {round(summary.Bankroll, 2)}"
                )
                self.ai.fit(self.old_matches, self.old_outcomes)
                self.last_retrain = month
                self.budget = summary.Bankroll

            self.update_models(games_increment)

        active_matches = cast(pd.DataFrame, opps[opps["Date"] == summary.Date])

        if active_matches.shape[0] == 0 or summary.Bankroll < (self.budget * 0.9):
            return pd.DataFrame(
                data=0,
                index=np.arange(active_matches.shape[0]),
                columns=pd.Index(["BetH", "BetA"], dtype="str"),
            )

        dataframe = self._create_dataframe(active_matches)
        probabilities = self.ai.get_probabilities(dataframe)
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

    def _get_match_parameters(self, match: Opp) -> pd.Series:
        """Get parameters for given match."""
        home_elo = self.elo.team_rating(match.HID)
        away_elo = self.elo.team_rating(match.AID)
        elo_by_location_prediction = self.elo_by_location.predict(match)

        rankings = pd.Series(
            [
                home_elo,
                away_elo,
                elo_by_location_prediction,
            ],
            index=self.RANKING_COLUMNS,
        )

        data_parameters = self.data.get_match_parameters(match)

        return pd.concat([rankings, data_parameters], axis=0)

    def train_ai_predictor(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""
        training_data = []
        outcomes_list = []

        for match in (Match(*x) for x in dataframe.itertuples()):
            match_parameters = self._get_match_parameters(match_to_opp(match))

            training_data.append(match_parameters)
            outcomes_list.append(match.HSC - match.ASC)

            self.data.add_match(match)
            self.elo.add_match(match)
            self.elo_by_location.add_match(match)

        training_dataframe = pd.DataFrame(
            training_data, columns=pd.Index(self.TRAINING_DATA_COLUMNS)
        )

        outcomes = pd.Series(outcomes_list)

        self.old_matches = training_dataframe
        self.old_outcomes = outcomes

        self.ai.fit(training_dataframe, outcomes)
        self.trained = True
