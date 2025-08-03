import os
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Configuration for the forecasting model."""
    features: List[str]
    target: str
    test_split_date: str
    random_state: int = 42
    cv_splits: int = 5
    
    @classmethod
    def default(cls):
        return cls(
            features=[
                'months_since_start', 'county_encoded', 'ev_total_lag1', 'ev_total_lag2', 
                'ev_total_lag3', 'ev_total_roll_mean_3', 'ev_total_pct_change_1', 
                'ev_total_pct_change_3', 'ev_percent', 'quarter', 'year'
            ],
            target='Electric Vehicle (EV) Total',
            test_split_date='2023-01-01'
        )