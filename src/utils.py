"""
Utility Functions for Traffic Demand Forecasting Application

This module provides:
- Data validation
- Formatting functions
- Statistical utilities
- Configuration helpers
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Road types in the dataset
ROAD_TYPES = ['Residential', 'Commercial', 'Highway']

# Weather conditions
WEATHER_CONDITIONS = ['Sunny', 'Rainy', 'Cloudy', 'Foggy']

# Large vehicles options
LARGE_VEHICLES_OPTIONS = ['Allowed', 'Not Allowed']

# Landmarks options
LANDMARKS_OPTIONS = ['Yes', 'No']


class DataValidator:
    """Validate input data for predictions."""
    
    @staticmethod
    def validate_hour(hour: int) -> bool:
        """Validate hour value (0-23)."""
        return isinstance(hour, int) and 0 <= hour <= 23
    
    @staticmethod
    def validate_minute(minute: int) -> bool:
        """Validate minute value (0-59)."""
        return isinstance(minute, int) and 0 <= minute <= 59
    
    @staticmethod
    def validate_day(day: int) -> bool:
        """Validate day value (1-7)."""
        return isinstance(day, int) and 1 <= day <= 7
    
    @staticmethod
    def validate_temperature(temp: float) -> bool:
        """Validate temperature (-50 to 60 Celsius)."""
        if temp is None:
            return True  # Optional
        return isinstance(temp, (int, float)) and -50 <= temp <= 60
    
    @staticmethod
    def validate_lanes(lanes: int) -> bool:
        """Validate number of lanes (1-8)."""
        return isinstance(lanes, int) and 1 <= lanes <= 8
    
    @staticmethod
    def validate_road_type(road_type: str) -> bool:
        """Validate road type."""
        return isinstance(road_type, str) and road_type.strip() != ""
    
    @staticmethod
    def validate_weather(weather: str) -> bool:
        """Validate weather condition."""
        return isinstance(weather, str) and weather.strip() != ""
    
    @staticmethod
    def validate_prediction_input(input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate all input fields for prediction.
        
        Args:
            input_data: Dictionary with prediction inputs
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            required_fields = [
                'geohash', 'day', 'RoadType', 'NumberofLanes',
                'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
                'hour', 'minute'
            ]
            
            # Check required fields
            for field in required_fields:
                if field not in input_data:
                    return False, f"Missing required field: {field}"
            
            # Validate specific fields
            if not DataValidator.validate_hour(input_data['hour']):
                return False, "Hour must be between 0 and 23"
            
            if not DataValidator.validate_minute(input_data['minute']):
                return False, "Minute must be between 0 and 59"
            
            if not DataValidator.validate_day(input_data['day']):
                return False, "Day must be between 1 and 7"
            
            if not DataValidator.validate_lanes(input_data['NumberofLanes']):
                return False, "Number of lanes must be between 1 and 8"
            
            if not DataValidator.validate_temperature(input_data.get('Temperature')):
                return False, "Temperature must be between -50 and 60°C"
            
            if not DataValidator.validate_road_type(input_data['RoadType']):
                return False, "Road type cannot be empty"
            
            if not DataValidator.validate_weather(input_data['Weather']):
                return False, "Weather cannot be empty"
            
            return True, "All inputs valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class Formatter:
    """Format output for display."""
    
    @staticmethod
    def format_demand(demand: float) -> str:
        """Format demand value as percentage."""
        return f"{demand * 100:.2f}%"
    
    @staticmethod
    def format_rmse(rmse: float) -> str:
        """Format RMSE metric."""
        return f"{rmse:.6f}"
    
    @staticmethod
    def format_r2(r2: float) -> str:
        """Format R² score."""
        return f"{r2:.4f}"
    
    @staticmethod
    def get_congestion_color(level: str) -> str:
        """Get color for congestion level."""
        colors = {
            'Low': '#00CC44',      # Green
            'Medium': '#FFAA00',   # Orange
            'High': '#CC0000'      # Red
        }
        return colors.get(level, '#808080')
    
    @staticmethod
    def get_congestion_emoji(level: str) -> str:
        """Get emoji for congestion level."""
        emojis = {
            'Low': '✅',
            'Medium': '⚠️',
            'High': '🚨'
        }
        return emojis.get(level, '❓')
    
    @staticmethod
    def format_time(hour: int, minute: int) -> str:
        """Format time as HH:MM."""
        return f"{hour:02d}:{minute:02d}"
    
    @staticmethod
    def format_day_name(day: int) -> str:
        """Convert day number to name."""
        days = {
            1: 'Monday',
            2: 'Tuesday',
            3: 'Wednesday',
            4: 'Thursday',
            5: 'Friday',
            6: 'Saturday',
            7: 'Sunday'
        }
        return days.get(day, f'Day {day}')
    
    @staticmethod
    def get_risk_color(demand_pct: float) -> str:
        """Get color based on demand percentage."""
        if demand_pct < 30:
            return '#00CC44'  # Green
        elif demand_pct < 70:
            return '#FFAA00'  # Orange
        else:
            return '#CC0000'  # Red


class StatisticsCalculator:
    """Calculate statistics for analytics."""
    
    @staticmethod
    def calculate_percentiles(data: np.ndarray, 
                             percentiles: List[int] = [10, 30, 50, 70, 90]) -> Dict[int, float]:
        """
        Calculate percentiles for data.
        
        Args:
            data: Numpy array of values
            percentiles: List of percentile values to calculate
        
        Returns:
            Dictionary mapping percentile to value
        """
        try:
            result = {}
            for p in percentiles:
                result[p] = float(np.percentile(data, p))
            return result
        except Exception as e:
            logger.error(f"Error calculating percentiles: {e}")
            return {}
    
    @staticmethod
    def calculate_demand_stats(demands: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for demand values.
        
        Args:
            demands: List of demand values
        
        Returns:
            Dictionary with min, max, mean, std, median
        """
        try:
            demands = np.array(demands)
            
            return {
                'min': float(np.min(demands)),
                'max': float(np.max(demands)),
                'mean': float(np.mean(demands)),
                'median': float(np.median(demands)),
                'std': float(np.std(demands)),
                'q25': float(np.percentile(demands, 25)),
                'q75': float(np.percentile(demands, 75))
            }
        except Exception as e:
            logger.error(f"Error calculating demand stats: {e}")
            return {}
    
    @staticmethod
    def get_peak_hours(demands_by_hour: Dict[int, float]) -> List[int]:
        """
        Get peak hours based on demand.
        
        Args:
            demands_by_hour: Dictionary mapping hour to demand
        
        Returns:
            List of peak hour numbers
        """
        try:
            if not demands_by_hour:
                return []
            
            mean_demand = np.mean(list(demands_by_hour.values()))
            peak_hours = [h for h, d in demands_by_hour.items() if d > mean_demand]
            return sorted(peak_hours)
        except Exception as e:
            logger.error(f"Error getting peak hours: {e}")
            return []


class DataGenerator:
    """Generate sample data for testing."""
    
    @staticmethod
    def generate_sample_input() -> Dict[str, Any]:
        """Generate a sample prediction input."""
        return {
            'geohash': 'qp02zt',
            'day': 3,
            'RoadType': 'Residential',
            'NumberofLanes': 3,
            'LargeVehicles': 'Allowed',
            'Landmarks': 'Yes',
            'Temperature': 28.5,
            'Weather': 'Sunny',
            'hour': 14,
            'minute': 30,
            'timestamp': '14:30'
        }
    
    @staticmethod
    def generate_sample_predictions(n: int = 24) -> Dict[int, float]:
        """
        Generate sample hourly predictions.
        
        Args:
            n: Number of hours
        
        Returns:
            Dictionary mapping hour to demand
        """
        np.random.seed(42)
        
        # Simulate realistic traffic pattern
        predictions = {}
        for hour in range(n):
            # Peak hours (8-10 AM and 5-7 PM)
            if hour in [8, 9, 17, 18]:
                base_demand = 0.7
            elif hour in [6, 7, 10, 16, 19]:
                base_demand = 0.5
            else:
                base_demand = 0.3
            
            # Add noise
            demand = np.clip(base_demand + np.random.normal(0, 0.1), 0, 1)
            predictions[hour] = float(demand)
        
        return predictions


def log_prediction(input_data: Dict[str, Any], 
                  prediction: Dict[str, Any]) -> None:
    """
    Log a prediction for auditing.
    
    Args:
        input_data: Input features
        prediction: Prediction result
    """
    try:
        logger.info(
            f"Prediction | Geohash: {input_data.get('geohash')} | "
            f"Time: {input_data.get('hour')}:{input_data.get('minute'):02d} | "
            f"Demand: {prediction.get('demand_percentage'):.2f}% | "
            f"Congestion: {prediction.get('congestion_level')}"
        )
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")