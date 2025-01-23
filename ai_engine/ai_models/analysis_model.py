import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisModel(ABC):
    """
    Abstract base class for all analysis models in the WAVE ecosystem.
    Defines the common interface and shared functionalities.
    """

    def __init__(self, config_path: str = 'config/analysis_model_config.json'):
        logger.info("Initializing AnalysisModel with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.model = None
        self.model_path = self.config.get('model_path', 'models/')
        os.makedirs(self.model_path, exist_ok=True)
        logger.info("AnalysisModel initialized successfully")

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data for the model.
        Should return features and labels.
        """
        pass

    @abstractmethod
    def build_model(self) -> Sequential:
        """
        Build and compile the TensorFlow/Keras model.
        """
        pass

    def train(self, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model with the provided data.
        """
        logger.info("Starting training for %s", self.__class__.__name__)
        X, y = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        self.model = self.build_model()
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        logger.info("Training completed for %s", self.__class__.__name__)
        return history.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        """
        if self.model is None:
            logger.error("Model is not trained yet.")
            raise ValueError("Model is not trained yet.")
        
        logger.info("Evaluating model: %s", self.__class__.__name__)
        predictions = self.model.predict(X_test)
        predictions = (predictions > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        logger.info("Evaluation results - Accuracy: %.4f, F1 Score: %.4f", accuracy, f1)
        return {'accuracy': accuracy, 'f1_score': f1}

    def save_model(self, filename: str = None):
        """
        Save the trained model to disk.
        """
        if self.model is None:
            logger.error("No model to save.")
            raise ValueError("No model to save.")
        
        filename = filename or f"{self.__class__.__name__}.h5"
        filepath = os.path.join(self.model_path, filename)
        self.model.save(filepath)
        logger.info("Model saved at %s", filepath)

    def load_model_from_disk(self, filename: str = None):
        """
        Load a trained model from disk.
        """
        filename = filename or f"{self.__class__.__name__}.h5"
        filepath = os.path.join(self.model_path, filename)
        if not os.path.exists(filepath):
            logger.error("Model file not found at %s", filepath)
            raise FileNotFoundError(f"Model file not found at {filepath}")
        self.model = load_model(filepath)
        logger.info("Model loaded from %s", filepath)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        if self.model is None:
            logger.error("Model is not loaded or trained.")
            raise ValueError("Model is not loaded or trained.")
        predictions = self.model.predict(X)
        return predictions

class CRISPRModel(AnalysisModel):
    """
    Model for optimizing CRISPR gene-editing strategies.
    """

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Loading data for CRISPRModel")
        data = self.db_connector.fetch_data('crispr_dataset')
        X = data.drop('efficiency', axis=1).values
        y = data['efficiency'].values
        logger.debug("CRISPRModel data loaded with shape: %s", X.shape)
        return X, y

    def build_model(self) -> Sequential:
        logger.info("Building CRISPRModel")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.config.get('input_dim'),)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.debug("CRISPRModel built and compiled")
        return model

class ProteinFoldingModel(AnalysisModel):
    """
    Model for predicting protein folding structures.
    """

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Loading data for ProteinFoldingModel")
        data = self.db_connector.fetch_data('protein_folding_dataset')
        X = data.drop('folding_accuracy', axis=1).values
        y = data['folding_accuracy'].values
        logger.debug("ProteinFoldingModel data loaded with shape: %s", X.shape)
        return X, y

    def build_model(self) -> Sequential:
        logger.info("Building ProteinFoldingModel")
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.config.get('input_dim'),)),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        logger.debug("ProteinFoldingModel built and compiled")
        return model

class EpigeneticModel(AnalysisModel):
    """
    Model for analyzing epigenetic modifications and their effects.
    """

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Loading data for EpigeneticModel")
        data = self.db_connector.fetch_data('epigenetic_dataset')
        X = data.drop('modification_effect', axis=1).values
        y = data['modification_effect'].values
        logger.debug("EpigeneticModel data loaded with shape: %s", X.shape)
        return X, y

    def build_model(self) -> Sequential:
        logger.info("Building EpigeneticModel")
        model = Sequential([
            Dense(200, activation='relu', input_shape=(self.config.get('input_dim'),)),
            Dropout(0.35),
            Dense(100, activation='relu'),
            Dropout(0.25),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.debug("EpigeneticModel built and compiled")
        return model

