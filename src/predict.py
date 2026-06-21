"""
Prediction Module for Traffic Demand Forecasting

This module handles:
- Loading trained models and feature engineer
- Making predictions on new data
- Ensemble prediction combination (0.8 LGB + 0.2 CatBoost)
- Traffic congestion classification
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import logging
import joblib
from pathlib import Path
from src.feature_engineering import FeatureEngineer
from src.model_loader import ModelLoader

logger = logging.getLogger(__name__)

# Congestion thresholds (0-100 percentile scale)
CONGESTION_THRESHOLDS = {
    'low': (0, 30),
    'medium': (30, 70),
    'high': (70, 100)
}


class TrafficPredictor:
    """
    Make traffic demand predictions and classify congestion levels.
    
    Attributes:
        lgb_model: Trained LightGBM model
        cat_model: Trained CatBoost model
        feature_engineer: Feature engineer instance
        demand_percentiles: Percentiles for congestion classification
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the predictor with trained models.
        
        Args:
            models_dir: Directory containing model files
        """
        self.lgb_model = None
        self.cat_model = None
        self.feature_engineer = None
        self.demand_percentiles = {}
        self.models_dir = models_dir
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load trained models and feature engineer from disk."""
        try:
            lgb_path = Path(self.models_dir) / 'lightgbm_model.pkl'
            cat_path = Path(self.models_dir) / 'catboost_model.pkl'
            fe_path = Path(self.models_dir) / 'feature_engineer.pkl'
            
            if lgb_path.exists():
                self.lgb_model = joblib.load(lgb_path)
                logger.info(f"Loaded LightGBM model from {lgb_path}")
            
            if cat_path.exists():
                self.cat_model = joblib.load(cat_path)
                logger.info(f"Loaded CatBoost model from {cat_path}")
            
            if fe_path.exists():
                self.feature_engineer = joblib.load(fe_path)
                logger.info(f"Loaded feature engineer from {fe_path}")
            
            if not (self.lgb_model and self.cat_model and self.feature_engineer):
                raise FileNotFoundError("One or more model files not found. Run train.py first.")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _prepare_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare user input for prediction.
        
        Args:
            input_data: Dictionary with user-provided features
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # Create DataFrame from input
            df = pd.DataFrame([input_data])
            
            # Fill missing values that might be in user input
            numeric_cols = ['Temperature', 'NumberofLanes']
            for col in numeric_cols:
                if col in df.columns and pd.isna(df[col].values[0]):
                    df[col] = 0
            
            categorical_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
            for col in categorical_cols:
                if col in df.columns and pd.isna(df[col].values[0]):
                    df[col] = 'Unknown'
            
            # Apply feature engineering
            df = self.feature_engineer.engineer_features(df, fit=False)
            
            # Drop unnecessary columns
            drop_cols = self.feature_engineer.get_drop_columns()
            df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
            
            # Get feature list
            feature_list = self.feature_engineer.get_feature_list()
            # Ensure all required features are present
            for col in feature_list:
                if col not in df.columns:
                    df[col] = 0
            
            return df[feature_list]
            
        except Exception as e:
            logger.error(f"Error preparing input: {e}")
            raise
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for traffic demand.
        
        Args:
            input_data: Dictionary with features:
                - geohash: str
                - day: int (1-7)
                - RoadType: str
                - NumberofLanes: int
                - LargeVehicles: str (Allowed/Not Allowed)
                - Landmarks: str (Yes/No)
                - Temperature: float
                - Weather: str (Sunny/Rainy/etc)
                - hour: int (0-23)
                - minute: int (0-59)
        
        Returns:
            Dictionary with prediction and congestion classification
        """
        try:
            # Prepare features
            X = self._prepare_input(input_data)
            
            # Make predictions
            lgb_pred = self.lgb_model.predict(X)[0]
            cat_pred = self.cat_model.predict(X)[0]
            
            # Ensemble prediction (0.8 LGB + 0.2 CatBoost)
            ensemble_pred = 0.8 * lgb_pred + 0.2 * cat_pred
            
            # Ensure prediction is in valid range
            ensemble_pred = np.clip(ensemble_pred, 0, 1)
            
            # Classify congestion level
            congestion_level = self._classify_congestion(ensemble_pred)
            
            result = {
                'predicted_demand': float(ensemble_pred),
                'demand_percentage': float(ensemble_pred * 100),
                'congestion_level': congestion_level,
                'lgb_prediction': float(lgb_pred),
                'cat_prediction': float(cat_pred),
                'ensemble_weight_lgb': 0.8,
                'ensemble_weight_cat': 0.2
            }
            
            logger.info(f"Prediction made: demand={ensemble_pred:.4f}, congestion={congestion_level}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def _classify_congestion(self, demand: float) -> str:
        """
        Classify traffic congestion level based on demand.
        
        Args:
            demand: Predicted demand value (0-1 scale)
            
        Returns:
            Congestion level: 'Low', 'Medium', or 'High'
        """
        # Convert to percentage
        demand_pct = demand * 100
        
        if demand_pct < 30:
            return 'Low'
        elif demand_pct < 70:
            return 'Medium'
        else:
            return 'High'
    
    def set_demand_percentiles(self, p30: float, p70: float) -> None:
        """
        Set custom percentile thresholds for congestion classification.
        
        Args:
            p30: 30th percentile value
            p70: 70th percentile value
        """
        self.demand_percentiles['p30'] = p30
        self.demand_percentiles['p70'] = p70
        logger.info(f"Percentile thresholds updated: p30={p30:.4f}, p70={p70:.4f}")
    
    def get_feature_importance(self) -> Tuple[list, list]:
        """
        Get feature importance from LightGBM model.
        
        Returns:
            Tuple of (feature_names, importance_values)
        """
        try:
            if self.lgb_model is None:
                raise ValueError("LightGBM model not loaded")
            
            importance = self.lgb_model.feature_importances_
            feature_names = self.lgb_model.feature_name_
            
            # Sort by importance
            sorted_idx = np.argsort(importance)[::-1]
            
            return (
                [feature_names[i] for i in sorted_idx],
                [importance[i] for i in sorted_idx]
            )
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise


def predict_traffic_demand(
    geohash: str,
    day: int,
    road_type: str,
    number_of_lanes: int,
    large_vehicles: str,
    landmarks: str,
    temperature: float,
    weather: str,
    hour: int,
    minute: int,
    models_dir: str = 'models'
) -> Dict[str, Any]:
    """
    Convenience function to make a single prediction.
    
    Args:
        geohash: Geographic hash
        day: Day of week (1-7)
        road_type: Type of road
        number_of_lanes: Number of lanes
        large_vehicles: Allowed/Not Allowed
        landmarks: Yes/No
        temperature: Temperature in Celsius
        weather: Weather condition
        hour: Hour (0-23)
        minute: Minute (0-59)
        models_dir: Directory with trained models
    
    Returns:
        Prediction dictionary
    """
    predictor = TrafficPredictor(models_dir)
    
    input_data = {
        'geohash': geohash,
        'day': day,
        'RoadType': road_type,
        'NumberofLanes': number_of_lanes,
        'LargeVehicles': large_vehicles,
        'Landmarks': landmarks,
        'Temperature': temperature,
        'Weather': weather,
        'hour': hour,
        'minute': minute,
        'timestamp': f'{hour}:{minute}'
    }
    
    return predictor.predict(input_data)
