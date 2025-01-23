import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BioEngineerModel(ABC):
    """
    Abstract base class for all bioengineering models in the WAVE ecosystem.
    Defines the common interface and shared functionalities.
    """

    def __init__(self, config_path: str = 'config/bioengineer_model_config.json'):
        logger.info("Initializing BioEngineerModel with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.model = None
        self.model_path = self.config.get('model_path', 'models/bioengineer/')
        os.makedirs(self.model_path, exist_ok=True)
        logger.info("BioEngineerModel initialized successfully")

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data for the model.
        Should return features and targets.
        """
        pass

    @abstractmethod
    def build_model(self) -> Sequential:
        """
        Build and compile the TensorFlow/Keras model.
        """
        pass

    def train(self, epochs: int = 200, batch_size: int = 64, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model with the provided data.
        """
        logger.info("Starting training for %s", self.__class__.__name__)
        X, y = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        self.model = self.build_model()

        checkpoint_path = os.path.join(self.model_path, f"{self.__class__.__name__}_best_model.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
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
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

        logger.info("Evaluation results - MAE: %.4f, MSE: %.4f", mae, mse)
        return {'mae': mae, 'mse': mse}

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

class GeneticPathwayOptimizationModel(BioEngineerModel):
    """
    Model for optimizing genetic pathways in bioengineering applications.
    """

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Loading data for GeneticPathwayOptimizationModel")
        data = self.db_connector.fetch_data('genetic_pathway_dataset')
        X = data.drop(['pathway_efficiency', 'pathway_success'], axis=1).values
        y = data['pathway_efficiency'].values
        logger.debug("GeneticPathwayOptimizationModel data loaded with shape: %s", X.shape)
        return X, y

    def build_model(self) -> Sequential:
        logger.info("Building GeneticPathwayOptimizationModel")
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.config.get('input_dim'),)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        logger.debug("GeneticPathwayOptimizationModel built and compiled")
        return model

class MetabolicEngineeringModel(BioEngineerModel):
    """
    Model for designing and optimizing metabolic pathways in bioengineering.
    """

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Loading data for MetabolicEngineeringModel")
        data = self.db_connector.fetch_data('metabolic_engineering_dataset')
        X = data.drop(['metabolic_output'], axis=1).values
        y = data['metabolic_output'].values
        logger.debug("MetabolicEngineeringModel data loaded with shape: %s", X.shape)
        return X, y

    def build_model(self) -> Sequential:
        logger.info("Building MetabolicEngineeringModel")
        model = Sequential([
            Dense(512, activation='relu', input_shape=(self.config.get('input_dim'),)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.0005)),
            loss='mean_absolute_error',
            metrics=['mae', 'mse']
        )
        logger.debug("MetabolicEngineeringModel built and compiled")
        return model

class EpigeneticModificationModel(BioEngineerModel):
    """
    Model for analyzing and predicting epigenetic modifications and their effects.
    """

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Loading data for EpigeneticModificationModel")
        data = self.db_connector.fetch_data('epigenetic_modification_dataset')
        X = data.drop(['modification_effectiveness'], axis=1).values
        y = data['modification_effectiveness'].values
        logger.debug("EpigeneticModificationModel data loaded with shape: %s", X.shape)
        return X, y

    def build_model(self) -> Sequential:
        logger.info("Building EpigeneticModificationModel")
        model = Sequential([
            Dense(300, activation='relu', input_shape=(self.config.get('input_dim'),)),
            BatchNormalization(),
            Dropout(0.35),
            Dense(150, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            Dense(75, activation='relu'),
            BatchNormalization(),
            Dropout(0.15),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.0008)),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.debug("EpigeneticModificationModel built and compiled")
        return model

