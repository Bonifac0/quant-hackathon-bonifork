from enum import IntEnum
from itertools import chain, product, repeat, starmap
from operator import add
from typing import TypeAlias

import numpy as np
import pandas as pd

from quant.types import Match

TeamID: TypeAlias = int


class Team(IntEnum):
    """Enum discerning teams playing home or away."""

    Home = 0
    Away = 1


class CustomQueue:  # TODO nahradit lepsi queue
    """Serve as custom version of queue."""

    def __init__(self, n: int) -> None:
        """Initialize queue."""
        self.size: int = n
        self.values: np.array = np.zeros((n, 1))
        self.__curent_oldest: int = 0

    def put(self, value: float) -> None:
        """Put new value in queue."""
        self.values[self.__curent_oldest % self.size] = value
        self.__curent_oldest += 1

    def get_q_avr(self) -> float:
        """Return average array of each feature."""
        if self.__curent_oldest == 0:
            return 0.0

        return (
            np.sum(self.values) / min(self.size, self.__curent_oldest)
            if self.__curent_oldest
            else 0.0
        )


class TeamData:
    """Hold data of one team, both as home and away."""

    N_SHORT = 5
    N_LONG = 30

    BASE_COLUMNS: tuple[str, ...] = (
        "WR",
        "WRH",
        "WRA",
        "PSA",
        "PSAH",
        "PSAA",
        "PLTA",
        "PLTAH",
        "PLTAA",
        "PD",
        "PDH",
        "PDA",
    )

    TEAM_COLUMNS: tuple[str, ...] = (
        "DSLM",
        *starmap(add, product(BASE_COLUMNS, ["_S", "_L"])),
    )

    # HACK: Python's scopes are weird, so we have to work around them with the
    # extra repeat iterator
    COLUMNS: tuple[tuple[str, ...], ...] = tuple(
        tuple(starmap(add, product(team_prefix, tc)))
        for team_prefix, tc in zip([["H_"], ["A_"]], repeat(TEAM_COLUMNS))
    )

    MATCH_COLUMNS: tuple[str, ...] = tuple(chain.from_iterable(COLUMNS))

    def __init__(self, team_id: TeamID) -> None:
        """Init datastucture."""
        self.id: TeamID = team_id
        self.date_last_match: pd.Timestamp = pd.to_datetime("1977-11-10")

        # short averages
        self.win_rate_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.win_rate_home_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.win_rate_away_S: CustomQueue = CustomQueue(TeamData.N_SHORT)

        self.points_scored_average_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.points_scored_average_home_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.points_scored_average_away_S: CustomQueue = CustomQueue(TeamData.N_SHORT)

        self.points_lost_to_x_average_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.points_lost_to_x_average_home_S: CustomQueue = CustomQueue(
            TeamData.N_SHORT
        )
        self.points_lost_to_x_average_away_S: CustomQueue = CustomQueue(
            TeamData.N_SHORT
        )

        self.points_diference_average_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.points_diference_average_home_S: CustomQueue = CustomQueue(
            TeamData.N_SHORT
        )
        self.points_diference_average_away_S: CustomQueue = CustomQueue(
            TeamData.N_SHORT
        )

        # long averages
        self.win_rate_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.win_rate_home_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.win_rate_away_L: CustomQueue = CustomQueue(TeamData.N_LONG)

        self.points_scored_average_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_scored_average_home_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_scored_average_away_L: CustomQueue = CustomQueue(TeamData.N_LONG)

        self.points_lost_to_x_average_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_lost_to_x_average_home_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_lost_to_x_average_away_L: CustomQueue = CustomQueue(TeamData.N_LONG)

        self.points_diference_average_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_diference_average_home_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_diference_average_away_L: CustomQueue = CustomQueue(TeamData.N_LONG)

    def _get_days_since_last_mach(self, today: pd.Timestamp) -> int:
        """Return number of days scince last mach."""
        return (today - self.date_last_match).days

    def update(self, match: Match, played_as: Team) -> None:
        """Update team data based on data from one mach."""
        self.date_last_match = pd.to_datetime(match.Date)

        win = match.H if played_as == Team.Home else match.A
        points = match.HSC if played_as == Team.Home else match.ASC
        points_lost_to = match.ASC if played_as == Team.Home else match.HSC
        point_diference = points - points_lost_to

        self.win_rate_S.put(win)
        self.win_rate_L.put(win)
        self.points_scored_average_S.put(points)
        self.points_scored_average_L.put(points)
        self.points_lost_to_x_average_S.put(points_lost_to)
        self.points_lost_to_x_average_L.put(points_lost_to)
        self.points_diference_average_S.put(point_diference)
        self.points_diference_average_L.put(point_diference)

        if played_as == Team.Home:
            self.win_rate_home_S.put(win)
            self.win_rate_home_L.put(win)
            self.points_scored_average_home_S.put(points)
            self.points_scored_average_home_L.put(points)
            self.points_lost_to_x_average_home_S.put(points_lost_to)
            self.points_lost_to_x_average_home_L.put(points_lost_to)
            self.points_diference_average_home_S.put(point_diference)
            self.points_diference_average_home_L.put(point_diference)
        else:
            self.win_rate_away_S.put(win)
            self.win_rate_away_L.put(win)
            self.points_scored_average_away_S.put(points)
            self.points_scored_average_away_L.put(points)
            self.points_lost_to_x_average_away_S.put(points_lost_to)
            self.points_lost_to_x_average_away_L.put(points_lost_to)
            self.points_diference_average_away_S.put(point_diference)
            self.points_diference_average_away_L.put(point_diference)

    def get_data_series(self, date: pd.Timestamp, team: Team) -> pd.Series:
        """Return complete data vector for given team."""
        return pd.Series(
            [
                self._get_days_since_last_mach(date),
                self.win_rate_S.get_q_avr(),
                self.win_rate_L.get_q_avr(),
                self.win_rate_home_S.get_q_avr(),
                self.win_rate_home_L.get_q_avr(),
                self.win_rate_away_S.get_q_avr(),
                self.win_rate_away_L.get_q_avr(),
                self.points_scored_average_S.get_q_avr(),
                self.points_scored_average_L.get_q_avr(),
                self.points_scored_average_home_S.get_q_avr(),
                self.points_scored_average_away_L.get_q_avr(),
                self.points_scored_average_home_L.get_q_avr(),
                self.points_scored_average_away_S.get_q_avr(),
                self.points_lost_to_x_average_S.get_q_avr(),
                self.points_lost_to_x_average_L.get_q_avr(),
                self.points_lost_to_x_average_home_S.get_q_avr(),
                self.points_lost_to_x_average_home_L.get_q_avr(),
                self.points_lost_to_x_average_away_S.get_q_avr(),
                self.points_lost_to_x_average_away_L.get_q_avr(),
                self.points_diference_average_S.get_q_avr(),
                self.points_diference_average_L.get_q_avr(),
                self.points_diference_average_home_S.get_q_avr(),
                self.points_diference_average_home_L.get_q_avr(),
                self.points_diference_average_away_S.get_q_avr(),
                self.points_diference_average_away_L.get_q_avr(),
            ],
            index=pd.Index(self.COLUMNS[team]),
        )
