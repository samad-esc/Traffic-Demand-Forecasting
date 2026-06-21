"""
Training Pipeline for Traffic Demand Forecasting

This module handles:
- Data loading and preprocessing
- Feature engineering
- Model training (LightGBM and CatBoost)
- Model evaluation and validation
- Model serialization
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Any
import warnings
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.feature_engineering import FeatureEngineer
from src.model_loader import ModelLoader

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
CONFIG = {
    'train_path': 'data/train.csv',
    'test_path': 'data/test.csv',
    'models_dir': 'models',
    'random_state': 42,
    'test_size': 0.2,
    'val_size': 0.2,
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'random_state': 42
    },
    'cat_params': {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 8,
        'random_state': 42,
        'verbose': 0,
        'early_stopping_rounds': 50
    }
}


class TrainingPipeline:
    """
    Complete training pipeline for traffic demand forecasting models.
    
    Attributes:
        config: Configuration dictionary
        feature_engineer: FeatureEngineer instance
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Target variables
        lgb_model: Trained LightGBM model
        cat_model: Trained CatBoost model
    """
    
    def __init__(self, config: dict = None):
        """Initialize the training pipeline."""
        self.config = config or CONFIG
        self.feature_engineer = FeatureEngineer()
        self.lgb_model = None
        self.cat_model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Create models directory if it doesn't exist
        Path(self.config['models_dir']).mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        try:
            logger.info(f"Loading training data from {self.config['train_path']}")
            train_df = pd.read_csv(self.config['train_path'])
            
            logger.info(f"Loading test data from {self.config['test_path']}")
            test_df = pd.read_csv(self.config['test_path'])
            
            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, 
                       train_df: pd.DataFrame, 
                       test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data: feature engineering and cleaning.
        
        Args:
            train_df: Raw training DataFrame
            test_df: Raw test DataFrame
            
        Returns:
            Tuple of (processed_train_df, processed_test_df)
        """
        try:
            logger.info("Starting feature engineering on training data")
            train_df = self.feature_engineer.engineer_features(train_df, fit=True)
            
            logger.info("Starting feature engineering on test data")
            test_df = self.feature_engineer.engineer_features(test_df, fit=False)
            
            # Drop unnecessary columns
            drop_cols = self.feature_engineer.get_drop_columns()
            train_df = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns])
            test_df = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])
            
            logger.info(f"Preprocessed train shape: {train_df.shape}")
            logger.info(f"Preprocessed test shape: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def prepare_train_val_test(self, 
                               train_df: pd.DataFrame, 
                               test_df: pd.DataFrame) -> None:
        """
        Prepare training, validation, and test sets.
        
        Args:
            train_df: Processed training DataFrame with target
            test_df: Processed test DataFrame
        """
        try:
            # Extract target variable
            y = train_df['demand'].values
            X = train_df.drop('demand', axis=1)
            
            # First split: train (80%) and temp (20%)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                X, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
            
            # Second split: split temp into val (50% of 20% = 10%) and test (50% of 20% = 10%)
            val_ratio = self.config['val_size']
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_temp, y_train_temp,
                test_size=val_ratio,
                random_state=self.config['random_state']
            )
            
            self.X_train = X_train
            self.X_val = X_val
            self.X_test = X_test_temp
            self.y_train = y_train
            self.y_val = y_val
            self.y_test = y_test_temp
            
            logger.info(f"Train set: {self.X_train.shape}")
            logger.info(f"Val set: {self.X_val.shape}")
            logger.info(f"Test set: {self.X_test.shape}")
            
        except Exception as e:
            logger.error(f"Error preparing train/val/test sets: {e}")
            raise
    
    def train_lightgbm(self) -> None:
        """Train LightGBM model."""
        try:
            logger.info("Starting LightGBM training")
    
            self.lgb_model = lgb.LGBMRegressor(
                **self.config['lgb_params']
            )
    
            self.lgb_model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                eval_metric='rmse'
            )
    
            # Evaluate
            y_pred_val = self.lgb_model.predict(
                self.X_val
            )
    
            rmse = np.sqrt(
                mean_squared_error(
                    self.y_val,
                    y_pred_val
                )
            )
    
            mae = mean_absolute_error(
                self.y_val,
                y_pred_val
            )
    
            r2 = r2_score(
                self.y_val,
                y_pred_val
            )
    
            logger.info(
                f"LightGBM - Val RMSE: {rmse:.6f}, "
                f"MAE: {mae:.6f}, "
                f"R²: {r2:.6f}"
            )
    
        except Exception as e:
            logger.error(
                f"Error training LightGBM: {e}"
            )
            raise
    
    def train_catboost(self) -> None:
        """Train CatBoost model."""
        try:
            logger.info("Starting CatBoost training")
            
            self.cat_model = CatBoostRegressor(**self.config['cat_params'])
            
            self.cat_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=0
            )
            
            # Evaluate
            y_pred_val = self.cat_model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred_val))
            mae = mean_absolute_error(self.y_val, y_pred_val)
            r2 = r2_score(self.y_val, y_pred_val)
            
            logger.info(f"CatBoost - Val RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            
        except Exception as e:
            logger.error(f"Error training CatBoost: {e}")
            raise
    
    def evaluate_ensemble(self) -> dict:
        """
        Evaluate ensemble model (0.8 * LGB + 0.2 * CatBoost).
        
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info("Evaluating ensemble model")
            
            y_pred_lgb = self.lgb_model.predict(self.X_test)
            y_pred_cat = self.cat_model.predict(self.X_test)
            
            # Ensemble prediction
            y_pred_ensemble = 0.8 * y_pred_lgb + 0.2 * y_pred_cat
            
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_ensemble))
            mae = mean_absolute_error(self.y_test, y_pred_ensemble)
            r2 = r2_score(self.y_test, y_pred_ensemble)
            
            metrics = {
                'ensemble_rmse': rmse,
                'ensemble_mae': mae,
                'ensemble_r2': r2,
                'y_pred': y_pred_ensemble,
                'y_true': self.y_test
            }
            
            logger.info(f"Ensemble - Test RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            raise
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        try:
            lgb_path = f"{self.config['models_dir']}/lightgbm_model.pkl"
            cat_path = f"{self.config['models_dir']}/catboost_model.pkl"
            
            joblib.dump(self.lgb_model, lgb_path)
            logger.info(f"LightGBM model saved to {lgb_path}")
            
            joblib.dump(self.cat_model, cat_path)
            logger.info(f"CatBoost model saved to {cat_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def save_feature_engineer(self) -> None:
        """Save feature engineer for later use."""
        try:
            fe_path = f"{self.config['models_dir']}/feature_engineer.pkl"
            joblib.dump(self.feature_engineer, fe_path)
            logger.info(f"Feature engineer saved to {fe_path}")
            
        except Exception as e:
            logger.error(f"Error saving feature engineer: {e}")
            raise
    
    def run_pipeline(self) -> dict:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary with metrics and results
        """
        logger.info("=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Load data
            train_df, test_df = self.load_data()
            
            # Preprocess
            train_df, test_df = self.preprocess_data(train_df, test_df)
            
            # Prepare sets
            self.prepare_train_val_test(train_df, test_df)
            
            # Train models
            self.train_lightgbm()
            self.train_catboost()
            
            # Evaluate
            metrics = self.evaluate_ensemble()
            
            # Save models
            self.save_models()
            self.save_feature_engineer()
            
            logger.info("=" * 80)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    metrics = pipeline.run_pipeline()
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS:")
    print(f"{'='*80}")
    print(f"Ensemble RMSE: {metrics['ensemble_rmse']:.6f}")
    print(f"Ensemble MAE: {metrics['ensemble_mae']:.6f}")
    print(f"Ensemble R² Score: {metrics['ensemble_r2']:.6f}")
    print(f"{'='*80}\n")
