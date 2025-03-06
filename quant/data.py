# extracts information from dataset
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, TypeAlias

import pandas as pd

from quant.data_helper import Team, TeamData

if TYPE_CHECKING:
    from quant.types import Match, Opp

TeamID: TypeAlias = int


class GamePlace(IntEnum):
    """Enum for game place."""

    Home = 0
    Away = 1
    Neutral = 2


class Data:
    """Class for working with data."""

    def __init__(self) -> None:
        """Create Data from csv file."""
        self.teams: dict[TeamID, TeamData] = {}

    def add_match(self, match: Match) -> None:
        """Update team data based on data from one mach."""
        self.teams.setdefault(match.HID, TeamData(match.HID)).update(match, Team.Home)
        self.teams.setdefault(match.AID, TeamData(match.AID)).update(match, Team.Away)

    def team_data(self, team_id: TeamID) -> TeamData:
        """Return the TeamData for given team."""
        return self.teams[team_id]

    def get_match_parameters(self, match: Opp) -> pd.Series:
        """Get array for match."""
        home_team = self.teams.setdefault(match.HID, TeamData(match.HID))
        away_team = self.teams.setdefault(match.AID, TeamData(match.AID))

        date: pd.Timestamp = pd.to_datetime(match.Date)

        return pd.concat(
            [
                home_team.get_data_series(date, Team.Home),
                away_team.get_data_series(date, Team.Away),
            ]
        )
