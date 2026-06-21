"""
Feature Engineering Module for Traffic Demand Forecasting

This module handles all feature transformations including:
- Temporal feature extraction (hour, minute, cyclical encoding)
- Interaction features (road-weather, geo-hour, etc.)
- Frequency-based features
- Missing value handling
- Categorical encoding
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles all feature engineering transformations for traffic demand prediction.
    
    Attributes:
        categorical_features: List of features requiring label encoding
        numeric_features: List of numeric features
        cyclical_features: Dict mapping features to their periods
    """
    
    def __init__(self):
        """Initialize feature engineer with feature lists."""
        self.categorical_features = [
            'geohash',
            'RoadType',
            'Weather',
            'LargeVehicles',
            'Landmarks',
            'road_weather',
            'geo_road'
        ]
        
        self.numeric_features = [
            'day',
            'NumberofLanes',
            'Temperature',
            'hour',
            'minute',
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'is_peak_hour',
            'rush_hour',
            'lane_hour',
            'lane_peak',
            'geo_freq'
        ]
        
        self.cyclical_features = {
            'hour': 24,
            'day': 7,
            'minute': 60
        }
        
        self.label_encoders: Dict = {}
        self.geo_freq_map: Dict = {}
        
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp and day columns.
        
        Args:
            df: DataFrame with 'timestamp' and 'day' columns
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        
        try:
            # Extract hour and minute from timestamp (format: "HH:MM")
            df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(int)
            
            # Define time slots based on hour
            df['time_slot'] = pd.cut(
                df['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                right=False,
                include_lowest=True
            )
            
            # Peak hours (typically 7-9 AM and 5-7 PM)
            df['is_peak_hour'] = ((df['hour'].isin([7, 8, 9, 17, 18, 19]))).astype(int)
            
            # Rush hours (broader definition)
            df['rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 21)).astype(int)
            
            logger.info("Temporal features extracted successfully")
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            raise
            
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical encoding for time features using sine/cosine transformation.
        
        Args:
            df: DataFrame with temporal features
            
        Returns:
            DataFrame with cyclical encoded features
        """
        df = df.copy()
        
        try:
            # Hour cyclical features (period = 24)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Day cyclical features (period = 7 for days of week)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
            
            logger.info("Cyclical features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating cyclical features: {e}")
            raise
            
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features combining different attributes.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        try:
            # Lane-hour interaction
            df['lane_hour'] = df['NumberofLanes'] * df['hour']
            
            # Lane-peak hour interaction
            df['lane_peak'] = df['NumberofLanes'] * df['is_peak_hour']
            
            # Road-weather interaction
            df['road_weather'] = (
                df['RoadType'].astype(str) + '_' + df['Weather'].astype(str)
            )
            
            # Geo-road interaction
            df['geo_road'] = (
                df['geohash'].astype(str) + '_' + df['RoadType'].astype(str)
            )
            
            logger.info("Interaction features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {e}")
            raise
            
        return df
    
    def create_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create frequency-based features (e.g., geohash frequency).
        
        Args:
            df: DataFrame with geohash column
            
        Returns:
            DataFrame with frequency features
        """
        df = df.copy()
        
        try:
            # Calculate geohash frequency (value counts)
            if not self.geo_freq_map:
                self.geo_freq_map = df['geohash'].value_counts().to_dict()
            
            df['geo_freq'] = df['geohash'].map(self.geo_freq_map).fillna(1)
            
            logger.info("Frequency features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating frequency features: {e}")
            raise
            
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in numeric and categorical columns.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        try:
            # Handle RoadType - fill with mode
            if df['RoadType'].isna().sum() > 0:
                road_type_mode = df['RoadType'].mode()[0] if not df['RoadType'].mode().empty else 'Unknown'
                df['RoadType'].fillna(road_type_mode, inplace=True)
                logger.info(f"Filled RoadType NaN with mode: {road_type_mode}")
            
            # Handle Temperature - fill with median
            if df['Temperature'].isna().sum() > 0:
                temp_median = df['Temperature'].median()
                df['Temperature'].fillna(temp_median, inplace=True)
                logger.info(f"Filled Temperature NaN with median: {temp_median:.2f}")
            
            # Handle Weather - fill with mode
            if df['Weather'].isna().sum() > 0:
                weather_mode = df['Weather'].mode()[0] if not df['Weather'].mode().empty else 'Sunny'
                df['Weather'].fillna(weather_mode, inplace=True)
                logger.info(f"Filled Weather NaN with mode: {weather_mode}")
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
            
        return df
    
    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.
        Handles unseen categories safely.
        """

        from sklearn.preprocessing import LabelEncoder

        df = df.copy()

        try:

            for col in self.categorical_features:

                if col not in df.columns:
                    continue

                df[col] = df[col].astype(str)

                if fit:

                    le = LabelEncoder()

                    le.fit(df[col])

                    df[col] = le.transform(df[col])

                    self.label_encoders[col] = le

                    logger.info(f"Fitted encoder for {col}")

                else:

                    if col not in self.label_encoders:

                        logger.warning(
                            f"No encoder found for {col}"
                        )
                        continue

                    le = self.label_encoders[col]

                    mapping = {
                        cls: idx
                        for idx, cls in enumerate(
                            le.classes_
                        )
                    }

                    df[col] = (
                        df[col]
                        .map(mapping)
                        .fillna(-1)
                        .astype(int)
                    )

                    logger.info(
                        f"Transformed {col}"
                    )

        except Exception as e:

            logger.error(
                f"Error encoding categorical features: {e}"
            )
            raise

        return df
    
    def engineer_features(self, 
                         df: pd.DataFrame, 
                         fit: bool = False) -> pd.DataFrame:
        """
        Main pipeline to engineer all features in sequence.
        
        Args:
            df: Raw DataFrame
            fit: Whether to fit encoders and other transformers
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        try:
            # Step 1: Handle missing values
            df = self.handle_missing_values(df)
            
            # Step 2: Extract temporal features
            df = self.extract_temporal_features(df)
            
            # Step 3: Create cyclical features
            df = self.create_cyclical_features(df)
            
            # Step 4: Create interaction features
            df = self.create_interaction_features(df)
            
            # Step 5: Create frequency features
            df = self.create_frequency_features(df)
            
            # Step 6: Encode categorical features
            df = self.encode_categorical_features(df, fit=fit)
            
            logger.info("Feature engineering pipeline completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {e}")
            raise
    
    def get_feature_list(self) -> List[str]:
        """
        Get list of final features for model input.
        
        Returns:
            List of feature names to use for modeling
        """
        features = [
            'geohash', 'day', 'RoadType', 'NumberofLanes',
            'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
            'hour', 'minute', 'is_peak_hour', 'rush_hour',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'lane_hour', 'lane_peak', 'road_weather', 'geo_road', 'geo_freq'
        ]
        return features
    
    def get_drop_columns(self) -> List[str]:
        """
        Get list of columns to drop before modeling.
        
        Returns:
            List of column names to drop
        """
        return ['Index', 'timestamp', 'time_slot']
