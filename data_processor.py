import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading, cleaning, and feature engineering."""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file with proper error handling."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            df = pd.read_csv(file_path, parse_dates=['Date'])
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        df_clean = df.copy()
        
        # Handle missing dates
        if df_clean['Date'].isnull().any():
            logger.warning("Found missing dates, dropping rows")
            df_clean = df_clean.dropna(subset=['Date'])
        
        # Convert string numbers with commas to float
        numeric_columns = ['Electric Vehicle (EV) Total', 'Total Vehicles']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .str.replace(',', '', regex=True)
                    .str.replace('$', '', regex=True)
                    .replace('nan', np.nan)
                    .astype(float)
                )
        
        # Remove rows with zero or negative values
        df_clean = df_clean[df_clean['Electric Vehicle (EV) Total'] > 0]
        df_clean = df_clean[df_clean['Total Vehicles'] > 0]
        
        logger.info(f"Data cleaned, shape: {df_clean.shape}")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for the model."""
        df_eng = df.copy()
        
        # Sort by county and date for proper time series feature creation
        df_eng = df_eng.sort_values(['County', 'Date'])
        
        # Time-based features
        df_eng['months_since_start'] = (df_eng['Date'] - df_eng['Date'].min()).dt.days // 30
        df_eng['quarter'] = df_eng['Date'].dt.quarter
        df_eng['year'] = df_eng['Date'].dt.year
        df_eng['month'] = df_eng['Date'].dt.month
        
        # Encode county
        if not self.is_fitted:
            df_eng['county_encoded'] = self.label_encoder.fit_transform(df_eng['County'])
            self.is_fitted = True
        else:
            df_eng['county_encoded'] = self.label_encoder.transform(df_eng['County'])
        
        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            df_eng[f'ev_total_lag{lag}'] = df_eng.groupby('County')['Electric Vehicle (EV) Total'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            df_eng[f'ev_total_roll_mean_{window}'] = (
                df_eng.groupby('County')['Electric Vehicle (EV) Total']
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            df_eng[f'ev_total_roll_std_{window}'] = (
                df_eng.groupby('County')['Electric Vehicle (EV) Total']
                .rolling(window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
        
        # Percentage change features
        for period in [1, 3, 6, 12]:
            df_eng[f'ev_total_pct_change_{period}'] = (
                df_eng.groupby('County')['Electric Vehicle (EV) Total'].pct_change(period)
            )
        
        # Ratio features
        df_eng['ev_percent'] = df_eng['Electric Vehicle (EV) Total'] / df_eng['Total Vehicles']
        df_eng['ev_growth_rate'] = df_eng.groupby('County')['ev_percent'].pct_change(1)
        
        # Market penetration features
        df_eng['market_saturation'] = np.minimum(df_eng['ev_percent'] * 100, 100)
        
        logger.info(f"Feature engineering completed, shape: {df_eng.shape}")
        return df_eng
    
    def split_data(self, df: pd.DataFrame, split_date: str, features: list[str], target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets based on date."""
        train_df = df[df['Date'] < split_date].copy()
        test_df = df[df['Date'] >= split_date].copy()
        
        # Clean infinite and NaN values
        for dataset in [train_df, test_df]:
            dataset[features] = dataset[features].replace([np.inf, -np.inf], np.nan)
            dataset.dropna(subset=features + [target], inplace=True)
        
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test