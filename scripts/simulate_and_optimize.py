# simulate_and_optimize.py

import pandas as pd
import numpy as np
from scipy.optimize import minimize

def apply_adstock(spend, decay):
    """
    Apply adstock transformation to account for lagged effects.
    """
    adstocked = []
    prev = 0
    for s in spend:
        new_val = s + decay * prev
        adstocked.append(new_val)
        prev = new_val
    return np.array(adstocked)

def simulate_scenario(df, channel, increase_pct, decay=0.5, coefficients=None):
    """
    Simulate increase in one channel and predict impact on sales.
    """
    df_copy = df.copy()
    df_copy[channel] *= (1 + increase_pct / 100)
    
    for col in ["TV_Spend", "Radio_Spend", "Digital_Spend", "Influencer_Spend", "OOH_Spend"]:
        df_copy[col + "_adstock"] = apply_adstock(df_copy[col], decay)

    if coefficients is None:
        raise ValueError("Provide coefficients for the model.")

    X = df_copy[[c + "_adstock" for c in coefficients.keys()]]
    sales_pred = np.dot(X, list(coefficients.values()))

    return sales_pred

def optimize_budget(df, total_budget, decay=0.5, coefficients=None):
    """
    Find optimal channel budget allocation that maximizes predicted sales.
    """
    if coefficients is None:
        raise ValueError("Provide coefficients for the model.")

    channels = list(coefficients.keys())

    def objective(weights):
        allocation = {channel: weight * total_budget for channel, weight in zip(channels, weights)}
        sales = 0
        for i, channel in enumerate(channels):
            adstocked = apply_adstock([allocation[channel]] * len(df), decay)
            sales += coefficients[channel] * np.sum(adstocked)
        return -sales  # maximize sales = minimize negative sales

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in channels]
    initial_weights = [1/len(channels)] * len(channels)

    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_allocation = {channel: weight * total_budget for channel, weight in zip(channels, result.x)}
        return optimal_allocation
    else:
        raise RuntimeError("Optimization failed.")
