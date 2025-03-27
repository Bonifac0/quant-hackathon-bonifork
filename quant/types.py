from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    import pandas as pd

TeamID: TypeAlias = int

Match = namedtuple(
    "Match",
    [
        "Index",
        "Season",
        "Date",
        "HID",
        "AID",
        "N",
        "POFF",
        "OddsH",
        "OddsA",
        "H",
        "A",
        "HSC",
        "ASC",
        "HFGM",
        "AFGM",
        "HFGA",
        "AFGA",
        "HFG3M",
        "AFG3M",
        "HFG3A",
        "AFG3A",
        "HFTM",
        "AFTM",
        "HFTA",
        "AFTA",
        "HORB",
        "AORB",
        "HDRB",
        "ADRB",
        "HRB",
        "ARB",
        "HAST",
        "AAST",
        "HSTL",
        "ASTL",
        "HBLK",
        "ABLK",
        "HTOV",
        "ATOV",
        "HPF",
        "APF",
    ],
    defaults=(None,) * 32,
)

Opp = namedtuple(
    "Opp",
    [
        "Index",
        "Season",
        "Date",
        "HID",
        "AID",
        "N",
        "POFF",
        "OddsH",
        "OddsA",
        "BetH",
        "BetA",
    ],
)

Summary = namedtuple(
    "Summary",
    [
        "Bankroll",
        "Date",
        "Min_bet",
        "Max_bet",
    ],
)


class RankingModel(Protocol):
    """Ranking model interface."""

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        raise NotImplementedError

    def rankings(self) -> dict[TeamID, float]:
        """Return normalized rankings."""
        raise NotImplementedError


class IAi(Protocol):
    """Ai interface."""

    def fit(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Check for implementation of fit fcn."""
        raise NotImplementedError("No fit fcn implemented.")

    def get_probabilities(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Check for implementation of get_propabilities fcn."""
        raise NotImplementedError("No get_propabilities fcn implemented.")


class IModel(Protocol):
    """Model interface."""

    def place_bets(
        self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame
    ) -> pd.DataFrame:
        """Check for implementation of bet fcn."""
        raise NotImplementedError("No place_bet fcn implemented.")
