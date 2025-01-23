import os
import json
import logging
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserInsightsManager:
    """
    Manages user insights within the WAVE ecosystem by analyzing user interactions,
    feedback, and behavior to generate actionable insights for improving AI models
    and platform features.
    """

    def __init__(self, config_path: str = 'config/user_insights_config.json'):
        logger.info("Initializing UserInsightsManager with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.insights_output = self.config.get('insights_output', 'reports/user_insights/')
        os.makedirs(self.insights_output, exist_ok=True)
        self.scaler = StandardScaler()
        logger.info("UserInsightsManager initialized successfully")

    def fetch_user_data(self) -> pd.DataFrame:
        """
        Fetch user interaction and feedback data from the database.
        """
        logger.info("Fetching user data for insights analysis")
        query = """
            SELECT 
                user_id, 
                session_id, 
                interaction_type, 
                interaction_timestamp, 
                feedback, 
                usage_duration, 
                feature_used
            FROM 
                user_interactions
            WHERE 
                interaction_timestamp >= %s
        """
        since_days = self.config.get('data_fetch', {}).get('since_days', 30)
        since_date = pd.Timestamp.now() - pd.Timedelta(days=since_days)
        try:
            data = self.db_connector.execute_query(query, (since_date,))
            df = pd.DataFrame(data)
            logger.info("Fetched %d user interaction records", len(df))
            return df
        except Exception as e:
            logger.error("Error fetching user data: %s", e)
            raise e

    def preprocess_user_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the user data by handling missing values, encoding categorical variables,
        and scaling numerical features.
        """
        logger.info("Preprocessing user data")
        try:
            # Handle missing values
            df['feedback'] = df['feedback'].fillna('')
            df['usage_duration'] = df['usage_duration'].fillna(df['usage_duration'].median())
            df['feature_used'] = df['feature_used'].fillna('Unknown')

            # Encode categorical variables
            df = pd.get_dummies(df, columns=['interaction_type', 'feature_used'], drop_first=True)

            # Scale numerical features
            numerical_features = ['usage_duration']
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])

            logger.debug("User data preprocessed with shape: %s", df.shape)
            return df
        except Exception as e:
            logger.error("Error preprocessing user data: %s", e)
            raise e

    def perform_clustering(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Perform K-Means clustering on user data to identify distinct user segments.
        """
        logger.info("Performing K-Means clustering with %d clusters", n_clusters)
        try:
            features = df.drop(['user_id', 'session_id', 'interaction_timestamp', 'feedback'], axis=1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(features)
            silhouette_avg = silhouette_score(features, df['cluster'])
            logger.info("K-Means clustering completed with silhouette score: %.4f", silhouette_avg)
            return df
        except Exception as e:
            logger.error("Error performing clustering: %s", e)
            raise e

    def analyze_feedback_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the sentiment of user feedback using TextBlob.
        """
        logger.info("Analyzing feedback sentiment")
        try:
            df['feedback_sentiment'] = df['feedback'].apply(self.get_sentiment)
            logger.debug("Feedback sentiment analysis completed")
            return df
        except Exception as e:
            logger.error("Error analyzing feedback sentiment: %s", e)
            raise e

    @staticmethod
    def get_sentiment(feedback: str) -> float:
        """
        Compute the sentiment polarity of the feedback text.
        """
        if feedback:
            return TextBlob(feedback).sentiment.polarity
        return 0.0

    def generate_user_segments_report(self, df: pd.DataFrame):
        """
        Generate and save a report of user segments based on clustering.
        """
        logger.info("Generating user segments report")
        try:
            segment_counts = df['cluster'].value_counts().sort_index()
            plt.figure(figsize=(10,6))
            sns.barplot(x=segment_counts.index, y=segment_counts.values, palette='viridis')
            plt.title('User Segments Distribution')
            plt.xlabel('Cluster')
            plt.ylabel('Number of Users')
            plt.tight_layout()
            plot_path = os.path.join(self.insights_output, 'user_segments_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info("User segments distribution plot saved at %s", plot_path)

            # Save segment counts to CSV
            segment_counts_df = segment_counts.reset_index()
            segment_counts_df.columns = ['Cluster', 'User_Count']
            segment_counts_df.to_csv(os.path.join(self.insights_output, 'user_segments_counts.csv'), index=False)
            logger.info("User segments counts saved at %s", os.path.join(self.insights_output, 'user_segments_counts.csv'))
        except Exception as e:
            logger.error("Error generating user segments report: %s", e)
            raise e

    def generate_feedback_sentiment_report(self, df: pd.DataFrame):
        """
        Generate and save a report of feedback sentiment distribution.
        """
        logger.info("Generating feedback sentiment report")
        try:
            plt.figure(figsize=(10,6))
            sns.histplot(df['feedback_sentiment'], bins=20, kde=True, color='blue')
            plt.title('Feedback Sentiment Distribution')
            plt.xlabel('Sentiment Polarity')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plot_path = os.path.join(self.insights_output, 'feedback_sentiment_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info("Feedback sentiment distribution plot saved at %s", plot_path)

            # Save sentiment statistics to JSON
            sentiment_stats = df['feedback_sentiment'].describe().to_dict()
            with open(os.path.join(self.insights_output, 'feedback_sentiment_stats.json'), 'w') as f:
                json.dump(sentiment_stats, f, indent=4)
            logger.info("Feedback sentiment statistics saved at %s", os.path.join(self.insights_output, 'feedback_sentiment_stats.json'))
        except Exception as e:
            logger.error("Error generating feedback sentiment report: %s", e)
            raise e

    def identify_top_features_used(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Identify the top N features used by users to inform feature prioritization.
        """
        logger.info("Identifying top %d features used by users", top_n)
        try:
            feature_columns = [col for col in df.columns if col.startswith('feature_used_')]
            feature_usage = df[feature_columns].sum().sort_values(ascending=False).head(top_n).reset_index()
            feature_usage.columns = ['Feature', 'Usage_Count']
            feature_usage.to_csv(os.path.join(self.insights_output, 'top_features_used.csv'), index=False)
            logger.info("Top features used saved at %s", os.path.join(self.insights_output, 'top_features_used.csv'))

            # Plot top features
            plt.figure(figsize=(12,8))
            sns.barplot(x='Usage_Count', y='Feature', data=feature_usage, palette='magma')
            plt.title(f'Top {top_n} Features Used by Users')
            plt.xlabel('Usage Count')
            plt.ylabel('Feature')
            plt.tight_layout()
            plot_path = os.path.join(self.insights_output, 'top_features_used.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info("Top features used plot saved at %s", plot_path)

            return feature_usage
        except Exception as e:
            logger.error("Error identifying top features used: %s", e)
            raise e

    def train_user_retention_model(self, df: pd.DataFrame):
        """
        Train a logistic regression model to predict user retention based on interaction data.
        """
        logger.info("Training user retention prediction model")
        try:
            # Assume 'retained' is a binary column indicating if the user is retained
            if 'retained' not in df.columns:
                logger.error("'retained' column not found in user data for retention prediction")
                raise ValueError("'retained' column not found in user data for retention prediction")

            X = df.drop(['user_id', 'session_id', 'interaction_timestamp', 'feedback', 'retained'], axis=1)
            y = df['retained']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            logger.info("User retention model trained successfully")

            # Evaluate the model
            predictions = model.predict(X_test)
            report = classification_report(y_test, predictions, output_dict=True)
            with open(os.path.join(self.insights_output, 'user_retention_model_report.json'), 'w') as f:
                json.dump(report, f, indent=4)
            logger.info("User retention model classification report saved at %s", os.path.join(self.insights_output, 'user_retention_model_report.json'))

            # Save the model parameters
            model_params = {
                'coefficients': model.coef_.tolist(),
                'intercept': model.intercept_.tolist(),
                'classes': model.classes_.tolist()
            }
            with open(os.path.join(self.insights_output, 'user_retention_model_params.json'), 'w') as f:
                json.dump(model_params, f, indent=4)
            logger.info("User retention model parameters saved at %s", os.path.join(self.insights_output, 'user_retention_model_params.json'))
        except Exception as e:
            logger.error("Error training user retention model: %s", e)
            raise e

    def generate_insights_dashboard(self):
        """
        Generate an interactive dashboard summarizing key user insights.
        """
        logger.info("Generating user insights dashboard")
        try:
            # This is a placeholder for dashboard generation logic.
            # Integration with tools like Dash or Streamlit can be implemented here.
            logger.info("User insights dashboard generation is not yet implemented.")
        except Exception as e:
            logger.error("Error generating user insights dashboard: %s", e)
            raise e

    def save_insights_to_database(self, insights: Dict[str, Any]):
        """
        Save generated insights to the database for future reference or integration.
        """
        logger.info("Saving user insights to the database")
        try:
            for key, df in insights.items():
                table_name = f'user_insights_{key}'
                self.db_connector.insert_dataframe(table_name, df)
                logger.debug("Saved insights '%s' to database table '%s'", key, table_name)
            logger.info("All user insights saved to the database successfully")
        except Exception as e:
            logger.error("Error saving user insights to the database: %s", e)
            raise e

    def run_insights_pipeline(self):
        """
        Execute the full user insights pipeline: fetch data, preprocess, analyze, generate reports,
        and save insights.
        """
        logger.info("Starting user insights pipeline")
        try:
            df = self.fetch_user_data()
            df = self.preprocess_user_data(df)
            df = self.perform_clustering(df)
            df = self.analyze_feedback_sentiment(df)

            # Generate reports
            self.generate_user_segments_report(df)
            self.generate_feedback_sentiment_report(df)
            self.identify_top_features_used(df)

            # Train retention model
            self.train_user_retention_model(df)

            # Save insights to database
            insights = {
                'user_segments': df[['user_id', 'cluster']],
                'feedback_sentiment': df[['user_id', 'feedback_sentiment']],
                'top_features_used': self.identify_top_features_used(df)
            }
            self.save_insights_to_database(insights)

            # Generate dashboard
            self.generate_insights_dashboard()

            logger.info("User insights pipeline completed successfully")
        except Exception as e:
            logger.error("User insights pipeline failed: %s", e)
            raise e

def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE User Insights Module')
    parser.add_argument('--config', type=str, default='config/user_insights_config.json', help='Path to user insights configuration file')
    args = parser.parse_args()

    try:
        manager = UserInsightsManager(config_path=args.config)
        manager.run_insights_pipeline()
    except Exception as e:
        logger.error("User insights analysis failed: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()

