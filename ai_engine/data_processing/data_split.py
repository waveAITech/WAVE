import os
import json
import logging
import sys
from typing import Any, Dict, Tuple, Optional, List

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSplitter:
    """
    Handles the splitting of datasets into training, validation, and testing sets.
    Supports various data types including genomic, clinical, and environmental data.
    """

    def __init__(self, config_path: str = 'config/data_split_config.json'):
        logger.info("Initializing DataSplitter with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.split_config = self.config.get('split_parameters', {})
        self.random_state = self.config.get('random_state', 42)
        logger.info("DataSplitter initialized successfully")

    def fetch_data(self, data_type: str) -> pd.DataFrame:
        """
        Fetches data from the database based on the data type.
        """
        logger.info("Fetching data for type: %s", data_type)
        query = self.split_config.get('queries', {}).get(data_type)
        if not query:
            logger.error("No query found for data type: %s", data_type)
            raise ValueError(f"No query found for data type: {data_type}")
        try:
            data = self.db_connector.execute_query(query)
            df = pd.DataFrame(data)
            logger.info("Fetched %d records for data type: %s", len(df), data_type)
            return df
        except SQLAlchemyError as e:
            logger.error("Database error while fetching data for %s: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Unexpected error while fetching data for %s: %s", data_type, e)
            raise e

    def preprocess_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Preprocesses the data by handling missing values, encoding categorical variables,
        and scaling numerical features.
        """
        logger.info("Preprocessing data for type: %s", data_type)
        try:
            # Handle missing values
            missing_strategy = self.split_config.get('preprocessing', {}).get('missing_values', {}).get(data_type, {})
            for column, strategy in missing_strategy.items():
                if strategy['method'] == 'fill':
                    if 'value' in strategy:
                        df[column] = df[column].fillna(strategy['value'])
                    elif 'method' in strategy:
                        df[column] = df[column].fillna(method=strategy['method'])
                elif strategy['method'] == 'drop':
                    df = df.dropna(subset=[column])
                logger.debug("Handled missing values for column: %s with strategy: %s", column, strategy['method'])

            # Encode categorical variables
            categorical_columns = self.split_config.get('preprocessing', {}).get('categorical', {}).get(data_type, [])
            for column in categorical_columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                logger.debug("Encoded categorical column: %s", column)

            # Scale numerical features
            numerical_columns = self.split_config.get('preprocessing', {}).get('numerical', {}).get(data_type, [])
            for column in numerical_columns:
                scaler_method = self.split_config.get('preprocessing', {}).get('scaling', {}).get('method', 'standard')
                if scaler_method == 'standard':
                    df[column] = (df[column] - df[column].mean()) / df[column].std()
                elif scaler_method == 'minmax':
                    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                logger.debug("Scaled numerical column: %s with method: %s", column, scaler_method)

            logger.info("Preprocessing completed for data type: %s", data_type)
            return df
        except KeyError as e:
            logger.error("Missing preprocessing configuration for %s: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Error during preprocessing for %s: %s", data_type, e)
            raise e

    def split_data(self, df: pd.DataFrame, data_type: str) -> Dict[str, pd.DataFrame]:
        """
        Splits the data into training, validation, and testing sets based on the configuration.
        Supports stratified splitting if specified.
        """
        logger.info("Splitting data for type: %s", data_type)
        try:
            split_params = self.split_config.get('splitting', {}).get(data_type, {})
            test_size = split_params.get('test_size', 0.2)
            validation_size = split_params.get('validation_size', 0.1)
            stratify_column = split_params.get('stratify_by')

            X = df.drop(split_params.get('target_column'), axis=1)
            y = df[split_params.get('target_column')]

            stratify = y if stratify_column and stratify_column in df.columns else None

            # First split into train+validation and test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
            )

            # Then split train_val into train and validation
            val_relative_size = validation_size / (1 - test_size)
            stratify_train_val = y_train_val if stratify_column and stratify_column in df.columns else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_relative_size, random_state=self.random_state, stratify=stratify_train_val
            )

            logger.info("Data split into train (%d), validation (%d), and test (%d)", len(X_train), len(X_val), len(X_test))

            return {
                'train': pd.concat([X_train, y_train], axis=1),
                'validation': pd.concat([X_val, y_val], axis=1),
                'test': pd.concat([X_test, y_test], axis=1)
            }
        except KeyError as e:
            logger.error("Missing splitting configuration for %s: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Error during data splitting for %s: %s", data_type, e)
            raise e

    def save_splits(self, splits: Dict[str, pd.DataFrame], data_type: str):
        """
        Saves the data splits to the database or files based on the configuration.
        """
        logger.info("Saving data splits for type: %s", data_type)
        try:
            save_params = self.split_config.get('save', {}).get(data_type, {})
            method = save_params.get('method', 'database')

            if method == 'database':
                tables = save_params.get('tables', {})
                for split_name, df_split in splits.items():
                    table_name = tables.get(split_name, f"{data_type}_{split_name}")
                    self.db_connector.insert_dataframe(table_name, df_split)
                    logger.info("Saved %s split to database table: %s", split_name, table_name)
            elif method == 'file':
                file_format = save_params.get('format', 'csv')
                output_dir = save_params.get('output_dir', f'data/splits/{data_type}/')
                os.makedirs(output_dir, exist_ok=True)
                for split_name, df_split in splits.items():
                    file_path = os.path.join(output_dir, f"{split_name}.{file_format}")
                    if file_format == 'csv':
                        df_split.to_csv(file_path, index=False)
                    elif file_format == 'json':
                        df_split.to_json(file_path, orient='records', lines=True)
                    logger.info("Saved %s split to file: %s", split_name, file_path)
            else:
                logger.error("Unsupported save method: %s", method)
                raise ValueError(f"Unsupported save method: {method}")

            logger.info("Data splits saved successfully for type: %s", data_type)
        except SQLAlchemyError as e:
            logger.error("Database error while saving splits for %s: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Error while saving splits for %s: %s", data_type, e)
            raise e

    def generate_split_report(self, splits: Dict[str, pd.DataFrame], data_type: str):
        """
        Generates a report summarizing the data splits.
        """
        logger.info("Generating split report for type: %s", data_type)
        try:
            report = {}
            for split_name, df_split in splits.items():
                report[split_name] = {
                    'record_count': len(df_split),
                    'features': df_split.drop(self.split_config.get('splitting', {}).get(data_type, {}).get('target_column'), axis=1).columns.tolist(),
                    'target': self.split_config.get('splitting', {}).get(data_type, {}).get('target_column')
                }

            report_path = self.split_config.get('report_output', f'reports/data_split/{data_type}_split_report.json')
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info("Data split report generated at %s", report_path)
        except Exception as e:
            logger.error("Error generating split report for %s: %s", data_type, e)
            raise e

    def run_split_pipeline(self, data_type: str):
        """
        Executes the full data splitting pipeline: fetch, preprocess, split, save, and report.
        """
        logger.info("Running data split pipeline for type: %s", data_type)
        try:
            df = self.fetch_data(data_type)
            df = self.preprocess_data(df, data_type)
            splits = self.split_data(df, data_type)
            self.save_splits(splits, data_type)
            self.generate_split_report(splits, data_type)
            logger.info("Data split pipeline completed successfully for type: %s", data_type)
        except Exception as e:
            logger.error("Data split pipeline failed for type %s: %s", data_type, e)
            raise e

    def run_all_pipelines(self):
        """
        Runs the data split pipeline for all configured data types.
        """
        logger.info("Running data split pipelines for all data types")
        data_types = self.split_config.get('data_types', [])
        for data_type in data_types:
            logger.info("Starting pipeline for data type: %s", data_type)
            self.run_split_pipeline(data_type)
        logger.info("All data split pipelines completed successfully")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Data Splitter Module')
    parser.add_argument('--config', type=str, default='config/data_split_config.json', help='Path to data split configuration file')
    parser.add_argument('--data_type', type=str, help='Specific data type to split')
    parser.add_argument('--all', action='store_true', help='Run split pipelines for all data types')
    args = parser.parse_args()

    splitter = DataSplitter(config_path=args.config)

    try:
        if args.all:
            splitter.run_all_pipelines()
        elif args.data_type:
            splitter.run_split_pipeline(args.data_type)
        else:
            logger.error("No action specified. Use --data_type or --all.")
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error("Data splitting failed: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()

