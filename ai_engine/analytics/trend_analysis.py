import os
import json
import logging
import sys
from typing import Any, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """
    Analyzes trends in genomic data and AI model performance within the WAVE ecosystem.
    Generates visualizations and reports for monitoring and decision-making.
    """

    def __init__(self, config_path: str = 'config/trend_analysis_config.json'):
        logger.info("Initializing TrendAnalyzer with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.report_output = self.config.get('report_output', 'reports/trend_analysis/')
        os.makedirs(self.report_output, exist_ok=True)
        logger.info("TrendAnalyzer initialized successfully")

    def fetch_genomic_data(self) -> pd.DataFrame:
        """
        Fetch genomic data from the database for trend analysis.
        """
        logger.info("Fetching genomic data for trend analysis")
        query = "SELECT gene_id, sequence, expression_level, modification_date FROM genomic_data"
        try:
            data = self.db_connector.execute_query(query)
            df = pd.DataFrame(data)
            logger.info("Fetched %d genomic data records", len(df))
            return df
        except Exception as e:
            logger.error("Error fetching genomic data: %s", e)
            raise e

    def fetch_model_performance(self) -> pd.DataFrame:
        """
        Fetch AI model performance metrics over time from the database.
        """
        logger.info("Fetching AI model performance data for trend analysis")
        query = """
            SELECT model_name, epoch, accuracy, loss, validation_accuracy, validation_loss, timestamp
            FROM model_performance
            ORDER BY timestamp ASC
        """
        try:
            data = self.db_connector.execute_query(query)
            df = pd.DataFrame(data)
            logger.info("Fetched %d model performance records", len(df))
            return df
        except Exception as e:
            logger.error("Error fetching model performance data: %s", e)
            raise e

    def analyze_genomic_trends(self, df_genomic: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trends in genomic data, such as gene expression levels over time.
        """
        logger.info("Analyzing genomic data trends")
        try:
            df_genomic['modification_date'] = pd.to_datetime(df_genomic['modification_date'])
            df_genomic.sort_values('modification_date', inplace=True)

            # Example trend: average expression level over time
            df_trend = df_genomic.groupby(pd.Grouper(key='modification_date', freq='M')).mean().reset_index()

            logger.info("Genomic trends analysis completed")
            return {'expression_trend': df_trend}
        except Exception as e:
            logger.error("Error analyzing genomic trends: %s", e)
            raise e

    def analyze_model_performance_trends(self, df_performance: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze AI model performance trends over epochs.
        """
        logger.info("Analyzing AI model performance trends")
        try:
            df_performance['timestamp'] = pd.to_datetime(df_performance['timestamp'])
            models = df_performance['model_name'].unique()
            trends = {}
            for model in models:
                df_model = df_performance[df_performance['model_name'] == model]
                trends[model] = df_model
                logger.debug("Processed performance data for model: %s", model)
            logger.info("Model performance trends analysis completed")
            return trends
        except Exception as e:
            logger.error("Error analyzing model performance trends: %s", e)
            raise e

    def visualize_genomic_trends(self, trends: Dict[str, Any]):
        """
        Generate and save visualizations for genomic data trends.
        """
        logger.info("Visualizing genomic data trends")
        try:
            df_trend = trends['expression_trend']
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='modification_date', y='expression_level', data=df_trend, marker='o')
            plt.title('Average Gene Expression Level Over Time')
            plt.xlabel('Modification Date')
            plt.ylabel('Average Expression Level')
            plt.tight_layout()
            plot_path = os.path.join(self.report_output, 'genomic_expression_trend.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info("Genomic expression trend plot saved at %s", plot_path)
        except Exception as e:
            logger.error("Error visualizing genomic trends: %s", e)
            raise e

    def visualize_model_performance_trends(self, trends: Dict[str, Any]):
        """
        Generate and save visualizations for AI model performance trends.
        """
        logger.info("Visualizing AI model performance trends")
        try:
            for model_name, df_model in trends.items():
                plt.figure(figsize=(12, 6))
                sns.lineplot(x='epoch', y='accuracy', data=df_model, label='Training Accuracy')
                sns.lineplot(x='epoch', y='validation_accuracy', data=df_model, label='Validation Accuracy')
                plt.title(f'Model Performance Over Epochs: {model_name}')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.tight_layout()
                plot_path = os.path.join(self.report_output, f'{model_name}_performance_trend.png')
                plt.savefig(plot_path)
                plt.close()
                logger.info("Model performance trend plot saved at %s", plot_path)
        except Exception as e:
            logger.error("Error visualizing model performance trends: %s", e)
            raise e

    def generate_report(self, genomic_trends: Dict[str, Any], model_trends: Dict[str, Any]):
        """
        Compile analysis results and visualizations into a comprehensive report.
        """
        logger.info("Generating trend analysis report")
        try:
            report_data = {
                'genomic_trends': genomic_trends,
                'model_performance_trends': model_trends
            }
            report_path = os.path.join(self.report_output, 'trend_analysis_report.json')
            with open(report_path, 'w') as report_file:
                json.dump(report_data, report_file, indent=4, default=str)
            logger.info("Trend analysis report saved at %s", report_path)
        except Exception as e:
            logger.error("Error generating trend analysis report: %s", e)
            raise e

    def run_trend_analysis(self):
        """
        Execute the full trend analysis pipeline: fetch data, analyze trends, visualize, and report.
        """
        logger.info("Starting trend analysis pipeline")
        try:
            df_genomic = self.fetch_genomic_data()
            df_performance = self.fetch_model_performance()

            genomic_trends = self.analyze_genomic_trends(df_genomic)
            model_trends = self.analyze_model_performance_trends(df_performance)

            self.visualize_genomic_trends(genomic_trends)
            self.visualize_model_performance_trends(model_trends)

            self.generate_report(genomic_trends, model_trends)
            logger.info("Trend analysis pipeline completed successfully")
        except Exception as e:
            logger.error("Trend analysis pipeline failed: %s", e)
            raise e

    def save_trend_data_to_database(self, genomic_trends: Dict[str, Any], model_trends: Dict[str, Any]):
        """
        Save analyzed trend data to the database for future reference or further analysis.
        """
        logger.info("Saving trend analysis data to the database")
        try:
            # Save genomic trends
            self.db_connector.insert_dataframe('genomic_trends', genomic_trends['expression_trend'])
            logger.debug("Genomic trends data saved to database")

            # Save model performance trends
            for model_name, df_model in model_trends.items():
                table_name = f'{model_name}_performance_trends'
                self.db_connector.insert_dataframe(table_name, df_model)
                logger.debug("Model performance trends for %s saved to database", model_name)

            logger.info("Trend analysis data saved to the database successfully")
        except Exception as e:
            logger.error("Error saving trend analysis data to the database: %s", e)
            raise e

    def visualize_transaction_trends(self):
        """
        Additional visualization: transaction trends from blockchain data.
        """
        logger.info("Visualizing transaction trends from blockchain data")
        try:
            # Fetch transaction volume and gas usage
            df_transactions = self.fetch_transaction_data()
            df_volume = self.analyze_transaction_volume(df_transactions)
            df_gas = self.analyze_gas_usage(df_transactions)

            # Plot transaction volume
            plt.figure(figsize=(12,6))
            sns.lineplot(x='blockNumber', y='transaction_count', data=df_volume, marker='o')
            plt.title('Transaction Volume Over Blocks')
            plt.xlabel('Block Number')
            plt.ylabel('Number of Transactions')
            plt.tight_layout()
            plot_path = os.path.join(self.report_output, 'transaction_volume_trend.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info("Transaction volume trend plot saved at %s", plot_path)

            # Plot gas usage
            plt.figure(figsize=(12,6))
            sns.lineplot(x='blockNumber', y='average_gas_price_gwei', data=df_gas, marker='o', color='orange')
            plt.title('Average Gas Price Over Blocks')
            plt.xlabel('Block Number')
            plt.ylabel('Average Gas Price (Gwei)')
            plt.tight_layout()
            plot_path = os.path.join(self.report_output, 'gas_usage_trend.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info("Gas usage trend plot saved at %s", plot_path)
        except Exception as e:
            logger.error("Error visualizing transaction trends: %s", e)
            raise e

    def fetch_transaction_data(self) -> pd.DataFrame:
        """
        Fetch transaction data from the database for trend visualization.
        """
        logger.info("Fetching transaction data for trend visualization")
        query = "SELECT blockNumber, gasPrice FROM transactions"
        try:
            data = self.db_connector.execute_query(query)
            df = pd.DataFrame(data)
            logger.info("Fetched %d transaction records", len(df))
            return df
        except Exception as e:
            logger.error("Error fetching transaction data: %s", e)
            raise e

    def analyze_transaction_volume(self, df_transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze transaction volume over blocks.
        """
        logger.info("Analyzing transaction volume over blocks")
        try:
            df_volume = df_transactions.groupby('blockNumber').size().reset_index(name='transaction_count')
            logger.debug("Transaction volume analysis completed")
            return df_volume
        except Exception as e:
            logger.error("Error analyzing transaction volume: %s", e)
            raise e

    def analyze_gas_usage(self, df_transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze average gas price over blocks.
        """
        logger.info("Analyzing average gas price over blocks")
        try:
            df_gas = df_transactions.groupby('blockNumber')['gasPrice'].mean().reset_index(name='average_gas_price_gwei')
            logger.debug("Gas usage analysis completed")
            return df_gas
        except Exception as e:
            logger.error("Error analyzing gas usage: %s", e)
            raise e

    def run_full_analysis(self):
        """
        Run the complete trend analysis including genomic data, model performance, and transaction trends.
        """
        logger.info("Running full trend analysis including genomic, model performance, and transaction data")
        try:
            self.run_trend_analysis()
            self.visualize_transaction_trends()
            # Optionally save trend data to database
            self.save_trend_data_to_database(
                self.analyze_genomic_trends(self.fetch_genomic_data()),
                self.analyze_model_performance_trends(self.fetch_model_performance())
            )
            logger.info("Full trend analysis completed successfully")
        except Exception as e:
            logger.error("Full trend analysis failed: %s", e)
            raise e

    def generate_summary_statistics(self):
        """
        Generate summary statistics for the trends and include them in the report.
        """
        logger.info("Generating summary statistics for trend analysis")
        try:
            df_genomic = self.fetch_genomic_data()
            df_performance = self.fetch_model_performance()

            # Genomic data summary
            genomic_summary = df_genomic.describe()

            # Model performance summary
            performance_summary = df_performance.groupby('model_name').agg({
                'accuracy': ['mean', 'std'],
                'loss': ['mean', 'std']
            }).reset_index()

            summary = {
                'genomic_summary': genomic_summary.to_dict(),
                'model_performance_summary': performance_summary.to_dict()
            }

            summary_path = os.path.join(self.report_output, 'summary_statistics.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            logger.info("Summary statistics saved at %s", summary_path)
        except Exception as e:
            logger.error("Error generating summary statistics: %s", e)
            raise e

def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Trend Analysis Module')
    parser.add_argument('--config', type=str, default='config/trend_analysis_config.json', help='Path to trend analysis configuration file')
    parser.add_argument('--full', action='store_true', help='Run full trend analysis including transaction trends')
    parser.add_argument('--save_db', action='store_true', help='Save analysis results to the database')
    parser.add_argument('--summary', action='store_true', help='Generate summary statistics for trend analysis')
    args = parser.parse_args()

    try:
        analyzer = TrendAnalyzer(config_path=args.config)
        if args.full:
            analyzer.run_full_analysis()
        else:
            analyzer.run_trend_analysis()
        if args.save_db:
            genomic_trends = analyzer.analyze_genomic_trends(analyzer.fetch_genomic_data())
            model_trends = analyzer.analyze_model_performance_trends(analyzer.fetch_model_performance())
            analyzer.save_trend_data_to_database(genomic_trends, model_trends)
        if args.summary:
            analyzer.generate_summary_statistics()
    except Exception as e:
        logger.error("Trend analysis failed: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()

