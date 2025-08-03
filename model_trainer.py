from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
from config import ModelConfig

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training, evaluation, and persistence."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.best_model = None
        self.model_performance = {}
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """Define parameter grids for different models."""
        return {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'RandomForest') -> Any:
        """Train model with hyperparameter tuning."""
        param_grids = self.get_param_grids()
        
        if model_type == 'RandomForest':
            base_model = RandomForestRegressor(random_state=self.config.random_state, n_jobs=-1)
        elif model_type == 'GradientBoosting':
            base_model = GradientBoostingRegressor(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grids[model_type],
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_iter=20,
            verbose=1,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        logger.info(f"Training {model_type} model...")
        search.fit(X_train, y_train)
        
        self.best_model = search.best_estimator_
        logger.info(f"Best parameters: {search.best_params_}")
        
        return search
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.best_model is None:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.best_model.predict(X_test)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, predictions),
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'R2': r2_score(y_test, predictions),
            'MAPE': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        self.model_performance = metrics
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.best_model is None:
            raise ValueError("No model to save")
        
        joblib.dump({
            'model': self.best_model,
            'config': self.config,
            'performance': self.model_performance
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.config = model_data['config']
        self.model_performance = model_data.get('performance', {})
        logger.info(f"Model loaded from {filepath}")