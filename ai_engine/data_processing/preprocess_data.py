import os
import json
import logging
import sys
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles the preprocessing of genomic, clinical, and environmental datasets within the WAVE ecosystem.
    Includes handling missing values, encoding categorical variables, scaling numerical features, 
    feature engineering, and dimensionality reduction.
    """

    def __init__(self, config_path: str = 'config/preprocess_data_config.json'):
        logger.info("Initializing DataPreprocessor with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        self.pca_transformers: Dict[str, Any] = {}
        self.pipeline_steps: Dict[str, Pipeline] = {}
        self.preprocessed_data: Dict[str, pd.DataFrame] = {}
        logger.info("DataPreprocessor initialized successfully")

    def fetch_raw_data(self, data_type: str) -> pd.DataFrame:
        """
        Fetch raw data from the database based on the data type.
        """
        logger.info("Fetching raw data for type: %s", data_type)
        query = self.preprocessing_config.get('queries', {}).get(data_type)
        if not query:
            logger.error("No query found for data type: %s", data_type)
            raise ValueError(f"No query found for data type: {data_type}")
        try:
            data = self.db_connector.execute_query(query)
            df = pd.DataFrame(data)
            logger.info("Fetched %d records for data type: %s", len(df), data_type)
            return df
        except Exception as e:
            logger.error("Error fetching raw data for %s: %s", data_type, e)
            raise e

    def handle_missing_values(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Handles missing values in the dataset based on the configuration.
        """
        logger.info("Handling missing values for data type: %s", data_type)
        try:
            missing_config = self.preprocessing_config.get('missing_values', {}).get(data_type, {})
            for column, strategy in missing_config.items():
                if strategy['method'] == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif strategy['method'] == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif strategy['method'] == 'most_frequent':
                    imputer = SimpleImputer(strategy='most_frequent')
                elif strategy['method'] == 'constant':
                    imputer = SimpleImputer(strategy='constant', fill_value=strategy.get('fill_value', ''))
                elif strategy['method'] == 'knn':
                    imputer = KNNImputer(n_neighbors=strategy.get('n_neighbors', 5))
                else:
                    logger.error("Unsupported imputation method: %s", strategy['method'])
                    raise ValueError(f"Unsupported imputation method: {strategy['method']}")
                
                df[[column]] = imputer.fit_transform(df[[column]])
                logger.debug("Applied %s imputation on column: %s", strategy['method'], column)
            logger.info("Missing values handled successfully for data type: %s", data_type)
            return df
        except KeyError as e:
            logger.error("Missing configuration for %s in missing_values: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Error handling missing values for %s: %s", data_type, e)
            raise e

    def encode_categorical_variables(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Encodes categorical variables using One-Hot Encoding or Label Encoding based on configuration.
        """
        logger.info("Encoding categorical variables for data type: %s", data_type)
        try:
            categorical_config = self.preprocessing_config.get('categorical_encoding', {}).get(data_type, {})
            for column, encoding_method in categorical_config.items():
                if encoding_method == 'onehot':
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[column]])
                    encoded_df = pd.DataFrame(encoded, columns=[f"{column}_{cat}" for cat in encoder.categories_[0]])
                    df = pd.concat([df.drop(column, axis=1), encoded_df], axis=1)
                    self.encoders[column] = encoder
                    logger.debug("Applied One-Hot Encoding on column: %s", column)
                elif encoding_method == 'label':
                    encoder = LabelEncoder()
                    df[column] = encoder.fit_transform(df[column].astype(str))
                    self.encoders[column] = encoder
                    logger.debug("Applied Label Encoding on column: %s", column)
                else:
                    logger.error("Unsupported encoding method: %s", encoding_method)
                    raise ValueError(f"Unsupported encoding method: {encoding_method}")
            logger.info("Categorical variables encoded successfully for data type: %s", data_type)
            return df
        except KeyError as e:
            logger.error("Missing configuration for %s in categorical_encoding: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Error encoding categorical variables for %s: %s", data_type, e)
            raise e

    def scale_numerical_features(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Scales numerical features using StandardScaler or MinMaxScaler based on configuration.
        """
        logger.info("Scaling numerical features for data type: %s", data_type)
        try:
            scaling_config = self.preprocessing_config.get('scaling', {}).get(data_type, {})
            for column, scaler_method in scaling_config.items():
                if scaler_method == 'standard':
                    scaler = StandardScaler()
                elif scaler_method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    logger.error("Unsupported scaling method: %s", scaler_method)
                    raise ValueError(f"Unsupported scaling method: {scaler_method}")
                
                df[[column]] = scaler.fit_transform(df[[column]])
                self.scalers[column] = scaler
                logger.debug("Applied %s scaling on column: %s", scaler_method, column)
            logger.info("Numerical features scaled successfully for data type: %s", data_type)
            return df
        except KeyError as e:
            logger.error("Missing configuration for %s in scaling: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Error scaling numerical features for %s: %s", data_type, e)
            raise e

    def feature_engineering(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Performs feature engineering tasks such as creating interaction terms, polynomial features, etc.
        """
        logger.info("Performing feature engineering for data type: %s", data_type)
        try:
            feature_eng_config = self.preprocessing_config.get('feature_engineering', {}).get(data_type, {})
            for new_feature, definition in feature_eng_config.items():
                if definition['type'] == 'interaction':
                    cols = definition['columns']
                    df[new_feature] = df[cols[0]] * df[cols[1]]
                    logger.debug("Created interaction feature: %s from columns: %s", new_feature, cols)
                elif definition['type'] == 'polynomial':
                    col = definition['column']
                    degree = definition.get('degree', 2)
                    df[f"{col}^{degree}"] = df[col] ** degree
                    logger.debug("Created polynomial feature: %s^%d", col, degree)
                elif definition['type'] == 'log':
                    col = definition['column']
                    df[f"log_{col}"] = np.log1p(df[col])
                    logger.debug("Created logarithmic feature: log_%s", col)
                else:
                    logger.error("Unsupported feature engineering type: %s", definition['type'])
                    raise ValueError(f"Unsupported feature engineering type: {definition['type']}")
            logger.info("Feature engineering completed successfully for data type: %s", data_type)
            return df
        except KeyError as e:
            logger.error("Missing configuration for %s in feature_engineering: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Error during feature engineering for %s: %s", data_type, e)
            raise e

    def dimensionality_reduction(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Reduces dimensionality of the dataset using PCA based on configuration.
        """
        logger.info("Performing dimensionality reduction for data type: %s", data_type)
        try:
            dr_config = self.preprocessing_config.get('dimensionality_reduction', {}).get(data_type, {})
            if dr_config.get('method') == 'pca':
                n_components = dr_config.get('n_components', 5)
                pca = PCA(n_components=n_components, random_state=self.config.get('random_state', 42))
                principal_components = pca.fit_transform(df)
                for i in range(n_components):
                    df[f'pca_{i+1}'] = principal_components[:, i]
                self.pca_transformers[data_type] = pca
                logger.debug("Applied PCA with %d components for data type: %s", n_components, data_type)
            else:
                logger.warning("No dimensionality reduction method specified for data type: %s", data_type)
            logger.info("Dimensionality reduction completed for data type: %s", data_type)
            return df
        except KeyError as e:
            logger.error("Missing configuration for %s in dimensionality_reduction: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Error during dimensionality reduction for %s: %s", data_type, e)
            raise e

    def select_features(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Selects the top K features based on feature selection techniques.
        """
        logger.info("Selecting features for data type: %s", data_type)
        try:
            feature_sel_config = self.preprocessing_config.get('feature_selection', {}).get(data_type, {})
            if feature_sel_config.get('method') == 'kbest':
                k = feature_sel_config.get('k', 10)
                selector = SelectKBest(score_func=f_classif, k=k)
                X = df.drop(feature_sel_config.get('target_column'), axis=1)
                y = df[feature_sel_config.get('target_column')]
                X_new = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support(indices=True)].tolist()
                df = pd.DataFrame(X_new, columns=selected_features)
                df[feature_sel_config.get('target_column')] = y.values
                self.feature_selectors[data_type] = selector
                logger.debug("Selected top %d features using KBest for data type: %s", k, data_type)
            else:
                logger.warning("No feature selection method specified for data type: %s", data_type)
            logger.info("Feature selection completed for data type: %s", data_type)
            return df
        except KeyError as e:
            logger.error("Missing configuration for %s in feature_selection: %s", data_type, e)
            raise e
        except Exception as e:
            logger.error("Error during feature selection for %s: %s", data_type, e)
            raise e

    def build_preprocessing_pipeline(self, data_type: str) -> Pipeline:
        """
        Constructs a preprocessing pipeline based on the configuration for the specified data type.
        """
        logger.info("Building preprocessing pipeline for data type: %s", data_type)
        try:
            pipeline_steps = []
            # Handle missing values
            missing_config = self.preprocessing_config.get('missing_values', {}).get(data_type, {})
            if missing_config:
                transformers = []
                for column, strategy in missing_config.items():
                    if strategy['method'] == 'mean':
                        transformers.append((f'impute_{column}', SimpleImputer(strategy='mean'), [column]))
                    elif strategy['method'] == 'median':
                        transformers.append((f'impute_{column}', SimpleImputer(strategy='median'), [column]))
                    elif strategy['method'] == 'most_frequent':
                        transformers.append((f'impute_{column}', SimpleImputer(strategy='most_frequent'), [column]))
                    elif strategy['method'] == 'constant':
                        transformers.append((f'impute_{column}', SimpleImputer(strategy='constant', fill_value=strategy.get('fill_value', '')), [column]))
                    elif strategy['method'] == 'knn':
                        transformers.append((f'impute_{column}', KNNImputer(n_neighbors=strategy.get('n_neighbors', 5)), [column]))
                if transformers:
                    pipeline_steps.append(('impute', ColumnTransformer(transformers, remainder='passthrough')))
                    logger.debug("Added missing value imputation to pipeline for data type: %s", data_type)
            
            # Encode categorical variables
            categorical_config = self.preprocessing_config.get('categorical_encoding', {}).get(data_type, {})
            if categorical_config:
                categorical_cols = list(categorical_config.keys())
                encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                pipeline_steps.append(('encode', ColumnTransformer([
                    ('onehot', encoder, categorical_cols)
                ], remainder='passthrough')))
                logger.debug("Added categorical encoding to pipeline for data type: %s", data_type)
            
            # Scale numerical features
            scaling_config = self.preprocessing_config.get('scaling', {}).get(data_type, {})
            if scaling_config:
                numerical_cols = list(scaling_config.keys())
                scaler = StandardScaler() if 'standard' in scaling_config.values() else MinMaxScaler()
                pipeline_steps.append(('scale', ColumnTransformer([
                    ('scaler', scaler, numerical_cols)
                ], remainder='passthrough')))
                logger.debug("Added scaling to pipeline for data type: %s", data_type)
            
            # Feature Engineering
            feature_eng_config = self.preprocessing_config.get('feature_engineering', {}).get(data_type, {})
            if feature_eng_config:
                # Custom transformers can be added here if needed
                logger.debug("Feature engineering steps added to pipeline for data type: %s", data_type)
            
            # Dimensionality Reduction
            dr_config = self.preprocessing_config.get('dimensionality_reduction', {}).get(data_type, {})
            if dr_config.get('method') == 'pca':
                pca = PCA(n_components=dr_config.get('n_components', 5), random_state=self.config.get('random_state', 42))
                pipeline_steps.append(('pca', pca))
                logger.debug("Added PCA to pipeline for data type: %s", data_type)
            
            # Feature Selection
            feature_sel_config = self.preprocessing_config.get('feature_selection', {}).get(data_type, {})
            if feature_sel_config.get('method') == 'kbest':
                selector = SelectKBest(score_func=f_classif, k=feature_sel_config.get('k', 10))
                pipeline_steps.append(('feature_selection', selector))
                logger.debug("Added feature selection to pipeline for data type: %s", data_type)
            
            pipeline = Pipeline(pipeline_steps)
            self.pipeline_steps[data_type] = pipeline
            logger.info("Preprocessing pipeline built successfully for data type: %s", data_type)
            return pipeline

    def apply_preprocessing_pipeline(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Applies the preprocessing pipeline to the dataset.
        """
        logger.info("Applying preprocessing pipeline for data type: %s", data_type)
        try:
            pipeline = self.build_preprocessing_pipeline(data_type)
            df_processed = pipeline.fit_transform(df)
            # Retrieve feature names after transformations
            processed_features = []
            for step in pipeline.steps:
                if isinstance(step[1], ColumnTransformer):
                    for name, transformer, columns in step[1].transformers_:
                        if name == 'remainder':
                            processed_features += columns
                        else:
                            if hasattr(transformer, 'get_feature_names_out'):
                                processed_features += list(transformer.get_feature_names_out(columns))
                            else:
                                processed_features += columns
                elif isinstance(step[1], PCA):
                    processed_features += [f'pca_{i+1}' for i in range(pipeline.named_steps['pca'].n_components)]
                elif isinstance(step[1], SelectKBest):
                    # Assuming feature names are preserved up to SelectKBest
                    processed_features += list(self.pipeline_steps[data_type].named_steps['feature_selection'].get_support(indices=True))
            df_processed = pd.DataFrame(df_processed, columns=processed_features)
            logger.info("Preprocessing pipeline applied successfully for data type: %s", data_type)
            return df_processed
        except Exception as e:
            logger.error("Error applying preprocessing pipeline for %s: %s", data_type, e)
            raise e

    def save_preprocessed_data(self, df: pd.DataFrame, data_type: str):
        """
        Saves the preprocessed data to the database or as files based on configuration.
        """
        logger.info("Saving preprocessed data for type: %s", data_type)
        try:
            save_config = self.preprocessing_config.get('save', {}).get(data_type, {})
            method = save_config.get('method', 'database')
            if method == 'database':
                table_name = save_config.get('table_name', f"{data_type}_preprocessed")
                self.db_connector.insert_dataframe(table_name, df)
                logger.info("Preprocessed data saved to database table: %s", table_name)
            elif method == 'file':
                file_format = save_config.get('format', 'csv')
                output_dir = save_config.get('output_dir', f'data/preprocessed/{data_type}/')
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f"preprocessed_{data_type}.{file_format}")
                if file_format == 'csv':
                    df.to_csv(file_path, index=False)
                elif file_format == 'json':
                    df.to_json(file_path, orient='records', lines=True)
                logger.info("Preprocessed data saved to file: %s", file_path)
            else:
                logger.error("Unsupported save method: %s", method)
                raise ValueError(f"Unsupported save method: {method}")
        except Exception as e:
            logger.error("Error saving preprocessed data for %s: %s", data_type, e)
            raise e

    def generate_preprocessing_report(self, data_type: str):
        """
        Generates a report summarizing the preprocessing steps and their outcomes.
        """
        logger.info("Generating preprocessing report for data type: %s", data_type)
        try:
            report = {
                'data_type': data_type,
                'preprocessing_steps': self.preprocessing_config.get('preprocessing', {}).get(data_type, {}),
                'encoders_used': list(self.encoders.keys()),
                'scalers_used': list(self.scalers.keys()),
                'feature_selectors_used': list(self.feature_selectors.keys()),
                'pca_transformers_used': list(self.pca_transformers.keys()),
                'pipeline_steps': list(self.pipeline_steps[data_type].named_steps.keys())
            }
            report_path = self.preprocessing_config.get('report_output', f'reports/preprocessing/{data_type}_preprocessing_report.json')
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info("Preprocessing report generated at %s", report_path)
        except Exception as e:
            logger.error("Error generating preprocessing report for %s: %s", data_type, e)
            raise e

    def run_preprocessing_pipeline(self, data_type: str):
        """
        Executes the full preprocessing pipeline: fetch data, handle missing values, encode categorical variables,
        scale numerical features, perform feature engineering, dimensionality reduction, feature selection,
        save preprocessed data, and generate reports.
        """
        logger.info("Running preprocessing pipeline for data type: %s", data_type)
        try:
            df_raw = self.fetch_raw_data(data_type)
            df_missing = self.handle_missing_values(df_raw, data_type)
            df_encoded = self.encode_categorical_variables(df_missing, data_type)
            df_scaled = self.scale_numerical_features(df_encoded, data_type)
            df_engineered = self.feature_engineering(df_scaled, data_type)
            df_reduced = self.dimensionality_reduction(df_engineered, data_type)
            df_selected = self.select_features(df_reduced, data_type)
            self.save_preprocessed_data(df_selected, data_type)
            self.generate_preprocessing_report(data_type)
            logger.info("Preprocessing pipeline completed successfully for data type: %s", data_type)
        except Exception as e:
            logger.error("Preprocessing pipeline failed for data type %s: %s", data_type, e)
            raise e

    def run_all_pipelines(self):
        """
        Runs the preprocessing pipeline for all configured data types.
        """
        logger.info("Running preprocessing pipelines for all data types")
        data_types = self.preprocessing_config.get('data_types', [])
        for data_type in data_types:
            logger.info("Starting preprocessing pipeline for data type: %s", data_type)
            self.run_preprocessing_pipeline(data_type)
        logger.info("All preprocessing pipelines completed successfully")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Data Preprocessing Module')
    parser.add_argument('--config', type=str, default='config/preprocess_data_config.json', help='Path to preprocessing configuration file')
    parser.add_argument('--data_type', type=str, help='Specific data type to preprocess')
    parser.add_argument('--all', action='store_true', help='Run preprocessing pipelines for all data types')
    args = parser.parse_args()

    preprocessor = DataPreprocessor(config_path=args.config)

    try:
        if args.all:
            preprocessor.run_all_pipelines()
        elif args.data_type:
            preprocessor.run_preprocessing_pipeline(args.data_type)
        else:
            logger.error("No action specified. Use --data_type or --all.")
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error("Data preprocessing failed: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()

