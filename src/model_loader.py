"""
Model Loader Module

This module provides utilities for:
- Saving and loading trained models
- Model versioning
- Model validation
- Serialization/deserialization
"""

import joblib
import pickle
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handle model persistence and loading operations.
    
    Supports both joblib and pickle serialization formats.
    """
    
    @staticmethod
    def save_model(model: Any, 
                   path: str, 
                   format: str = 'joblib',
                   compress: bool = True) -> bool:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model object
            path: File path to save to
            format: Serialization format ('joblib' or 'pickle')
            compress: Whether to compress (joblib only)
        
        Returns:
            bool: True if successful
        """
        try:
            # Create parent directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'joblib':
                joblib.dump(model, path, compress=3 if compress else 0)
                logger.info(f"Model saved to {path} using joblib")
            elif format == 'pickle':
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model saved to {path} using pickle")
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @staticmethod
    def load_model(path: str, 
                   format: str = 'joblib') -> Optional[Any]:
        """
        Load a trained model from disk.
        
        Args:
            path: File path to load from
            format: Serialization format ('joblib' or 'pickle')
        
        Returns:
            Loaded model object or None if failed
        """
        try:
            if not Path(path).exists():
                logger.error(f"Model file not found: {path}")
                return None
            
            if format == 'joblib':
                model = joblib.load(path)
                logger.info(f"Model loaded from {path} using joblib")
            elif format == 'pickle':
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Model loaded from {path} using pickle")
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def save_multiple_models(models: dict, 
                            base_dir: str = 'models',
                            format: str = 'joblib') -> bool:
        """
        Save multiple models to a directory.
        
        Args:
            models: Dictionary of {name: model}
            base_dir: Base directory for models
            format: Serialization format
        
        Returns:
            bool: True if all successful
        """
        try:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            
            all_successful = True
            for name, model in models.items():
                path = f"{base_dir}/{name}.pkl"
                success = ModelLoader.save_model(model, path, format)
                all_successful = all_successful and success
            
            return all_successful
            
        except Exception as e:
            logger.error(f"Error saving multiple models: {e}")
            return False
    
    @staticmethod
    def load_multiple_models(base_dir: str = 'models',
                            model_names: list = None,
                            format: str = 'joblib') -> dict:
        """
        Load multiple models from a directory.
        
        Args:
            base_dir: Base directory containing models
            model_names: List of model names (without extension)
            format: Serialization format
        
        Returns:
            Dictionary of {name: model}
        """
        try:
            models = {}
            
            if model_names is None:
                # Load all .pkl files in directory
                model_paths = Path(base_dir).glob('*.pkl')
                model_names = [p.stem for p in model_paths]
            
            for name in model_names:
                path = f"{base_dir}/{name}.pkl"
                model = ModelLoader.load_model(path, format)
                if model is not None:
                    models[name] = model
            
            logger.info(f"Loaded {len(models)} models from {base_dir}")
            return models
            
        except Exception as e:
            logger.error(f"Error loading multiple models: {e}")
            return {}


class ModelVersionManager:
    """
    Manage model versions with timestamps and metadata.
    """
    
    def __init__(self, base_dir: str = 'models/versions'):
        """Initialize version manager."""
        self.base_dir = base_dir
        Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    def save_versioned_model(self, 
                            model: Any, 
                            model_name: str,
                            metrics: dict = None,
                            format: str = 'joblib') -> str:
        """
        Save a model with timestamp version.
        
        Args:
            model: Model to save
            model_name: Name of model (e.g., 'lightgbm')
            metrics: Optional dictionary with evaluation metrics
            format: Serialization format
        
        Returns:
            Version string (timestamp)
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_dir = f"{self.base_dir}/{model_name}/{timestamp}"
            Path(version_dir).mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = f"{version_dir}/model.pkl"
            ModelLoader.save_model(model, model_path, format)
            
            # Save metadata
            if metrics:
                metadata_path = f"{version_dir}/metrics.json"
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Metrics saved to {metadata_path}")
            
            logger.info(f"Model version {timestamp} saved for {model_name}")
            return timestamp
            
        except Exception as e:
            logger.error(f"Error saving versioned model: {e}")
            return None
    
    def load_latest_model(self, model_name: str, format: str = 'joblib') -> Any:
        """
        Load the latest version of a model.
        
        Args:
            model_name: Name of model
            format: Serialization format
        
        Returns:
            Latest model or None if not found
        """
        try:
            model_dir = Path(self.base_dir) / model_name
            
            if not model_dir.exists():
                logger.error(f"No versions found for {model_name}")
                return None
            
            # Get latest directory (sorted by name)
            versions = sorted(model_dir.iterdir())
            if not versions:
                return None
            
            latest_path = versions[-1] / 'model.pkl'
            model = ModelLoader.load_model(str(latest_path), format)
            
            logger.info(f"Loaded latest version of {model_name}: {versions[-1].name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading latest model: {e}")
            return None
    
    def get_model_versions(self, model_name: str) -> list:
        """
        Get list of available versions for a model.
        
        Args:
            model_name: Name of model
        
        Returns:
            List of version timestamps
        """
        try:
            model_dir = Path(self.base_dir) / model_name
            
            if not model_dir.exists():
                return []
            
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            return versions
            
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []


class ModelValidator:
    """
    Validate models for correctness and compatibility.
    """
    
    @staticmethod
    def validate_model(model: Any, 
                      required_methods: list = None) -> bool:
        """
        Validate that a model has required methods.
        
        Args:
            model: Model to validate
            required_methods: List of required method names
        
        Returns:
            bool: True if valid
        """
        try:
            if required_methods is None:
                required_methods = ['predict', 'fit']
            
            for method in required_methods:
                if not hasattr(model, method):
                    logger.warning(f"Model missing method: {method}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False
    
    @staticmethod
    def validate_predictions(predictions: list,
                            min_val: float = 0.0,
                            max_val: float = 1.0) -> bool:
        """
        Validate that predictions are in expected range.
        
        Args:
            predictions: List of predictions
            min_val: Minimum expected value
            max_val: Maximum expected value
        
        Returns:
            bool: True if all predictions in range
        """
        try:
            import numpy as np
            
            predictions = np.array(predictions)
            
            if np.any(predictions < min_val) or np.any(predictions > max_val):
                logger.warning(f"Predictions outside range [{min_val}, {max_val}]")
                return False
            
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                logger.warning("Predictions contain NaN or Inf values")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating predictions: {e}")
            return False
