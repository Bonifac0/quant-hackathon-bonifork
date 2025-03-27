import numpy as np
import pandas as pd

from quant.types import Match, Opp


# called from predictors.py
def sigmoid(score_difference: float) -> float:
    """Calculate the probability of home team winning based on score difference."""
    slope = 0.2  # range optimal 0.1 to 1. liked 0.3 and 0.5 (maybe 1)
    return 1 / (1 + np.exp(-slope * score_difference))


# called from predictors.py
def calculate_probabilities(score_differences: np.ndarray) -> pd.DataFrame:
    """Calculate the probabilities of teams winning based on score differences."""
    probabilities = []

    for score_difference in score_differences:
        home_prob = sigmoid(score_difference)
        away_prob = 1 - home_prob
        probabilities.append((home_prob, away_prob))

    return pd.DataFrame(probabilities, columns=pd.Index(["WinHome", "WinAway"]))


# called from model.py
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
