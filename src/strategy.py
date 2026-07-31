from __future__ import annotations
import pandas as pd

def calculate_12_1_momentum(monthly_close: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate prior-(2,12) momentum for each ETF.

    For a row dated month t:
        momentum[t] = price[t - 1] / price[t - 12] - 1
    This excludes the current month and uses the previous
    11 complete monthly returns.

    Parameters
    ----------
    monthly_close:
        DataFrame indexed by month-end dates.
        Each column represents one ETF.
        Each value is the adjusted month-end closing price.

    Returns
    -------
    pd.DataFrame
        Momentum scores with the same index and columns.
    """

    if monthly_close.empty():
        raise ValueError("monthly_close cannot be empty.")
    if not isinstance(monthly_close.index, pd.DatetimeIndex):
        raise TypeError("monthly_close must use a pandas DateTimeIndex")

    momentum = monthly_close.shift(1) / monthly_close.shift(12) - 1
    return momentum

def select_top_assets(momentum_scores: pd.Series, number_of_assets: int = 5, require_positive_momentum: bool = False) -> pd.Series:
    """
    Select the highest-momentum ETFs for one formation month

    ----------
    momentum_scores:
        Series indexed by ticker symbol containing momentum scores.

    number_of_assets:
        Maximum number of ETFs to select.

    require_positive_momentum:
        When True, ETFs with momentum <= 0 are excluded.

    Returns
    -------
    pd.Series
        Momentum scores for the selected ETFs, ordered from
        highest to lowest.
    """

    if number_of_assets <= 0:
        raise ValueError("number_of_assets must be a positive integer.")

    scores = momentum_scores.dropna()
    if requires_positive_momentum:
        scores = scores[scores > 0]
    return scores.nlargest(number_of_assets)

def calculate_equal_weights(momentum_scores: pd.Series, number_of_assets: int = 5,
                            require_positive_momentum: bool = False, use_fixed_slots: bool = True) -> pd.Series:
    """
    Calculate target portfolio weights for one formation month.

    Pure cross-sectional strategy:
        Select the top ETFs regardless of whether momentum is positive.

    Filtered strategy:
        Select only ETFs with positive momentum.

    Fixed-slot weighting:
        Each selected ETF receives 1 / number_of_assets.
        Unused capital remains as cash.

    Redistributed weighting:
        Capital is divided equally among however many ETFs qualify.

    Parameters
    ----------
    momentum_scores:
        Series indexed by ETF ticker.

    number_of_assets:
        Maximum number of ETFs held.

    require_positive_momentum:
        Exclude ETFs whose momentum is zero or negative.

    use_fixed_slots:
        If True, each position receives 1 / number_of_assets.
        If False, capital is divided among all selected ETFs.

    Returns
    -------
    pd.Series
        Target ETF weights. Unselected assets receive weight zero.
    """

    selected = select_top_assets(momentum_scores=momentum_scores, number_of_assets=number_of_assets, 
                                require_positive_momentum=require_positive_momentum)
    weights = pd.Series(0.0, index=momentum_scores.index, dtype=float)

    if selected.empty:
        return weights
    if use_fixed_slots:
        position.weight = 1.0 / number_of_assets
    else:
        position.weight = 1.0 / len(selected)

    weights.loc[selected.index] = position.weight
    return weights

def generate_target_weights(monthly_close: pd.DataFrame, number_of_assets: int = 5, 
                            require_positive_momentum: bool = False, use_fixed_slots: bool = True) -> pd.DataFrame:
    """"
    Generate target ETF weights for every month.

    The weights at month t are based on information available
    before or at month t and are intended to be applied to the
    following month's returns.

    Returns
    -------
    pd.DataFrame
        Target weights indexed by formation month.
    """

    momentum = calculate_12_1_momentum(monthly_close)
    weights = momentum.apply(lambda row: calculate_equal_weights(
        momentum_scores=row, number_of_assets=number_of_assets, 
        require_positive_momentum=require_positive_momentum, use_fixed_slots=use_fixed_slots), axis=1)
    return weights

def calculate_cash_weights(asset_weights: pd.DataFrame) -> pd.Series:
    """
    Calculate the portion of the portfolio held in cash.

    Cash weight is:
        1 - sum of ETF weights
    """

    cash_weights = 1.0 - asset_weights.sum(axis=1)
    return cash_weights.clip(lower=0.0, upper=1.0)