import pandas as pd
import numpy as np
from typing import Dict, List
from config import ModelConfig
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import logging

logger = logging.getLogger(__name__)

class EVForecaster:
    """Main forecasting pipeline."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig.default()
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer(self.config)
        self.raw_data = None
        self.processed_data = None
    
    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Complete data loading and preparation pipeline."""
        # Load data
        self.raw_data = self.data_processor.load_data(file_path)
        
        # Clean and engineer features
        cleaned_data = self.data_processor.clean_data(self.raw_data)
        self.processed_data = self.data_processor.engineer_features(cleaned_data)
        
        return self.processed_data
    
    def train_and_evaluate(self, model_type: str = 'RandomForest') -> Dict[str, float]:
        """Train model and return evaluation metrics."""
        if self.processed_data is None:
            raise ValueError("Data must be loaded and prepared first")
        
        # Split data
        X_train, X_test, y_train, y_test = self.data_processor.split_data(
            self.processed_data, 
            self.config.test_split_date, 
            self.config.features, 
            self.config.target
        )
        
        # Train model
        self.model_trainer.train_model(X_train, y_train, model_type)
        
        # Evaluate
        metrics = self.model_trainer.evaluate_model(X_test, y_test)
        
        return metrics
    
    def predict_future(self, months_ahead: int = 12) -> pd.DataFrame:
        """Generate future predictions."""
        if self.model_trainer.best_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Get latest data for each county
        latest_data = self.processed_data.loc[
            self.processed_data.groupby('County')['Date'].idxmax()
        ].copy()
        
        predictions = []
        
        for _, row in latest_data.iterrows():
            county_predictions = []
            current_row = row.copy()
            
            for month in range(1, months_ahead + 1):
                # Update time features
                future_date = current_row['Date'] + pd.DateOffset(months=month)
                current_row['months_since_start'] += 1
                current_row['quarter'] = future_date.quarter
                current_row['year'] = future_date.year
                current_row['month'] = future_date.month
                
                # Make prediction
                features_subset = [f for f in self.config.features if f in current_row.index]
                X_pred = current_row[features_subset].values.reshape(1, -1)
                
                # Handle missing features
                if len(features_subset) < len(self.config.features):
                    X_pred_full = np.zeros((1, len(self.config.features)))
                    for i, feature in enumerate(self.config.features):
                        if feature in features_subset:
                            idx = features_subset.index(feature)
                            X_pred_full[0, i] = X_pred[0, idx]
                    X_pred = X_pred_full
                
                prediction = self.model_trainer.best_model.predict(X_pred)[0]
                
                county_predictions.append({
                    'County': current_row['County'],
                    'Date': future_date,
                    'Predicted_EV_Total': max(0, prediction),
                    'Months_Ahead': month
                })
                
                # Update lagged features for next iteration
                if month == 1:
                    current_row['ev_total_lag1'] = current_row[self.config.target]
                
            predictions.extend(county_predictions)
        
        return pd.DataFrame(predictions)