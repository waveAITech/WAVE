import logging
import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, config_path: str = 'config/data_cleaner_config.json'):
        logger.debug("Initializing DataCleaner with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.anomaly_detector = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        self.scaler = StandardScaler()
        logger.debug("DataCleaner initialized successfully")

    def load_data(self, source: str) -> pd.DataFrame:
        logger.debug("Loading data from source: %s", source)
        if source.endswith('.csv'):
            data = pd.read_csv(source)
        elif source.endswith('.json'):
            data = pd.read_json(source)
        else:
            logger.error("Unsupported file format for source: %s", source)
            raise ValueError("Unsupported file format")
        logger.debug("Data loaded successfully with shape: %s", data.shape)
        return data

    def validate_data(self, data: pd.DataFrame, data_type: str) -> bool:
        logger.debug("Validating data for type: %s", data_type)
        validation_rules = self.config.get('validation_rules').get(data_type, {})
        for column, rules in validation_rules.items():
            if column not in data.columns:
                logger.error("Missing required column: %s", column)
                return False
            if 'dtype' in rules:
                if not pd.api.types.is_dtype_equal(data[column].dtype, np.dtype(rules['dtype'])):
                    logger.error("Data type mismatch for column: %s. Expected: %s, Found: %s",
                                 column, rules['dtype'], data[column].dtype)
                    return False
            if 'regex' in rules:
                pattern = re.compile(rules['regex'])
                if not data[column].astype(str).str.match(pattern).all():
                    logger.error("Regex validation failed for column: %s", column)
                    return False
            if 'range' in rules:
                min_val, max_val = rules['range']
                if not data[column].between(min_val, max_val).all():
                    logger.error("Range validation failed for column: %s. Expected between %s and %s",
                                 column, min_val, max_val)
                    return False
        logger.debug("Data validation passed for type: %s", data_type)
        return True

    def clean_genomic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Cleaning genomic data")
        # Remove duplicates
        initial_shape = data.shape
        data = data.drop_duplicates(subset=['gene_id', 'sequence'])
        logger.debug("Removed duplicates: %s -> %s", initial_shape, data.shape)

        # Fill missing values
        data['sequence'] = data['sequence'].fillna(method='ffill')
        logger.debug("Filled missing values in 'sequence' column")

        # Standardize gene sequences
        data['sequence'] = data['sequence'].str.upper().str.replace(r'[^ATCG]', '', regex=True)
        logger.debug("Standardized gene sequences")

        # Validate data
        if not self.validate_data(data, 'genomic'):
            logger.error("Genomic data validation failed")
            raise ValueError("Genomic data validation failed")
        
        logger.debug("Genomic data cleaned successfully")
        return data

    def clean_clinical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Cleaning clinical data")
        # Handle missing values
        data = data.fillna({
            'age': data['age'].median(),
            'gender': 'Unknown',
            'diagnosis': 'Undiagnosed'
        })
        logger.debug("Filled missing values in clinical data")

        # Remove outliers in age using Z-score
        data = data[(np.abs(data['age'] - data['age'].mean()) <= (3 * data['age'].std()))]
        logger.debug("Removed outliers in 'age' column")

        # Encode categorical variables
        data['gender'] = data['gender'].astype('category').cat.codes
        logger.debug("Encoded 'gender' column")

        # Validate data
        if not self.validate_data(data, 'clinical'):
            logger.error("Clinical data validation failed")
            raise ValueError("Clinical data validation failed")
        
        logger.debug("Clinical data cleaned successfully")
        return data

    def clean_environmental_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Cleaning environmental data")
        # Impute missing soil quality with mean
        data['soil_quality'] = data['soil_quality'].fillna(data['soil_quality'].mean())
        logger.debug("Imputed missing 'soil_quality' values")

        # Normalize environmental parameters
        numeric_cols = ['soil_quality', 'pH', 'temperature']
        data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        logger.debug("Normalized environmental parameters: %s", numeric_cols)

        # Validate data
        if not self.validate_data(data, 'environmental'):
            logger.error("Environmental data validation failed")
            raise ValueError("Environmental data validation failed")
        
        logger.debug("Environmental data cleaned successfully")
        return data

    def detect_anomalies(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        logger.debug("Detecting anomalies in data")
        self.anomaly_detector.fit(data[features])
        predictions = self.anomaly_detector.predict(data[features])
        anomalies = data[predictions == -1]
        logger.debug("Detected %d anomalies", anomalies.shape[0])
        return anomalies

    def integrate_external_data(self, data: pd.DataFrame, external_source: str) -> pd.DataFrame:
        logger.debug("Integrating external data from source: %s", external_source)
        external_data = self.api_client.fetch_data(external_source)
        data = pd.concat([data, external_data], ignore_index=True)
        logger.debug("Integrated external data. New shape: %s", data.shape)
        return data

    def save_clean_data(self, data: pd.DataFrame, destination: str):
        logger.debug("Saving cleaned data to destination: %s", destination)
        if destination.endswith('.csv'):
            data.to_csv(destination, index=False)
        elif destination.endswith('.json'):
            data.to_json(destination, orient='records', lines=True)
        else:
            logger.error("Unsupported file format for destination: %s", destination)
            raise ValueError("Unsupported file format")
        logger.debug("Cleaned data saved successfully")

    def run_cleaning_pipeline(self, source: str, destination: str, data_type: str):
        logger.debug("Running cleaning pipeline for data type: %s", data_type)
        data = self.load_data(source)
        
        if data_type == 'genomic':
            cleaned_data = self.clean_genomic_data(data)
        elif data_type == 'clinical':
            cleaned_data = self.clean_clinical_data(data)
        elif data_type == 'environmental':
            cleaned_data = self.clean_environmental_data(data)
        else:
            logger.error("Unsupported data type: %s", data_type)
            raise ValueError("Unsupported data type")
        
        # Detect anomalies
        features = self.config.get('anomaly_detection').get(data_type, {}).get('features', data.columns.tolist())
        anomalies = self.detect_anomalies(cleaned_data, features)
        if not anomalies.empty:
            logger.warning("Anomalies detected in the data. Check logs for details.")
            # Optionally handle anomalies (e.g., remove, flag, notify)
        
        # Integrate external data if specified
        external_sources = self.config.get('external_data_sources').get(data_type, [])
        for ext_source in external_sources:
            cleaned_data = self.integrate_external_data(cleaned_data, ext_source)
        
        # Save cleaned data
        self.save_clean_data(cleaned_data, destination)
        logger.debug("Cleaning pipeline completed for data type: %s", data_type)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Data Cleaning Module')
    parser.add_argument('--source', type=str, required=True, help='Path to the input data file (CSV or JSON)')
    parser.add_argument('--destination', type=str, required=True, help='Path to save the cleaned data file (CSV or JSON)')
    parser.add_argument('--data_type', type=str, required=True, choices=['genomic', 'clinical', 'environmental'], help='Type of data to clean')
    
    args = parser.parse_args()
    
    cleaner = DataCleaner()
    cleaner.run_cleaning_pipeline(args.source, args.destination, args.data_type)

