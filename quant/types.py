from __future__ import annotations  # noqa: A005

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

    def place_bets(
        self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame
    ) -> pd.DataFrame:
        """Check for implementation of bet fcn."""
        raise NotImplementedError


class IModel(Protocol):
    """Model interface."""

    def place_bets(
        self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame
    ) -> pd.DataFrame:
        """Check for implementation of bet fcn."""
        raise NotImplementedError


def match_to_opp(match: Match) -> Opp:
    """
    Convert Match to Opp.

    Fills Bets with 0.
    """
    return Opp(
        Index=match.Index,
        Season=match.Season,
        Date=match.Date,
        HID=match.HID,
        AID=match.AID,
        N=match.N,
        POFF=match.POFF,
        OddsH=match.OddsH,
        OddsA=match.OddsA,
        BetH=0,
        BetA=0,
    )
