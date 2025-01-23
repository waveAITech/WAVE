import os
import json
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient
from wave_ai.models import CRISPRModel, ProteinFoldingModel, EpigeneticModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenomeStrategy(ABC):
    """
    Abstract base class for all genome editing strategies within the WAVE ecosystem.
    Defines the interface and shared functionalities for strategy generation and evaluation.
    """

    def __init__(self, config_path: str = 'config/genome_strategies_config.json'):
        logger.info("Initializing GenomeStrategy with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.strategy_params = self.config.get('strategy_parameters', {})
        self.output_dir = self.config.get('output_dir', 'strategies/')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("GenomeStrategy initialized successfully")

    @abstractmethod
    def generate_strategy(self, target_gene: str, desired_trait: str) -> Dict[str, Any]:
        """
        Generates a genome editing strategy for the specified target gene and desired trait.
        """
        pass

    @abstractmethod
    def evaluate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the effectiveness and safety of the given genome editing strategy.
        """
        pass

    @abstractmethod
    def optimize_strategy(self, target_gene: str, desired_trait: str) -> Dict[str, Any]:
        """
        Optimizes the genome editing strategy using evolutionary algorithms or other optimization techniques.
        """
        pass

    def save_strategy(self, strategy: Dict[str, Any], filename: str):
        """
        Saves the genome editing strategy to a JSON file and stores it in the database.
        """
        try:
            file_path = os.path.join(self.output_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(strategy, f, indent=4)
            logger.info("Strategy saved to file: %s", file_path)

            # Store in database
            self.db_connector.insert_record('genome_strategies', strategy)
            logger.info("Strategy stored in the database successfully")
        except Exception as e:
            logger.error("Error saving strategy: %s", e)
            raise e

    def load_existing_strategies(self) -> pd.DataFrame:
        """
        Loads existing genome editing strategies from the database.
        """
        logger.info("Loading existing genome editing strategies from the database")
        try:
            df_strategies = self.db_connector.fetch_dataframe('genome_strategies')
            logger.info("Loaded %d existing strategies", len(df_strategies))
            return df_strategies
        except SQLAlchemyError as e:
            logger.error("Database error while loading strategies: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error while loading strategies: %s", e)
            raise e


class CRISPRStrategy(GenomeStrategy):
    """
    Implements genome editing strategies using CRISPR-Cas9 technology.
    """

    def __init__(self, config_path: str = 'config/genome_strategies_config.json'):
        super().__init__(config_path)
        self.model = CRISPRModel()
        logger.info("CRISPRStrategy initialized successfully")

    def generate_strategy(self, target_gene: str, desired_trait: str) -> Dict[str, Any]:
        logger.info("Generating CRISPR strategy for gene: %s to achieve trait: %s", target_gene, desired_trait)
        try:
            # Fetch gene sequence from database or external source
            gene_sequence = self.fetch_gene_sequence(target_gene)
            if not gene_sequence:
                logger.error("Gene sequence for %s not found", target_gene)
                raise ValueError(f"Gene sequence for {target_gene} not found")

            # Identify potential sgRNA targets
            sgRNAs = self.identify_sgRNAs(gene_sequence)

            # Generate strategy dictionary
            strategy = {
                'strategy_type': 'CRISPR-Cas9',
                'target_gene': target_gene,
                'desired_trait': desired_trait,
                'sgRNAs': sgRNAs,
                'gene_sequence': gene_sequence,
                'generated_at': pd.Timestamp.now().isoformat()
            }

            logger.info("CRISPR strategy generated successfully for gene: %s", target_gene)
            return strategy
        except Exception as e:
            logger.error("Error generating CRISPR strategy: %s", e)
            raise e

    def evaluate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Evaluating CRISPR strategy for gene: %s", strategy['target_gene'])
        try:
            # Use AI model to predict efficiency and off-target effects
            efficiency_scores = self.model.predict_efficiency(strategy['sgRNAs'])
            off_target_scores = self.model.predict_off_target(strategy['sgRNAs'])

            # Aggregate evaluation metrics
            evaluation = {
                'strategy_type': strategy['strategy_type'],
                'target_gene': strategy['target_gene'],
                'desired_trait': strategy['desired_trait'],
                'efficiency_scores': efficiency_scores,
                'off_target_scores': off_target_scores,
                'evaluation_metrics': {
                    'average_efficiency': np.mean(efficiency_scores),
                    'max_off_target': np.max(off_target_scores)
                },
                'evaluated_at': pd.Timestamp.now().isoformat()
            }

            logger.info("CRISPR strategy evaluated successfully for gene: %s", strategy['target_gene'])
            return evaluation
        except Exception as e:
            logger.error("Error evaluating CRISPR strategy: %s", e)
            raise e

    def optimize_strategy(self, target_gene: str, desired_trait: str) -> Dict[str, Any]:
        logger.info("Optimizing CRISPR strategy for gene: %s to achieve trait: %s", target_gene, desired_trait)
        try:
            # Differential Evolution optimization to maximize efficiency and minimize off-target effects
            bounds = [(0, 1) for _ in range(len(self.identify_sgRNAs(self.fetch_gene_sequence(target_gene))))]
            
            def objective(x):
                strategy = self.generate_strategy(target_gene, desired_trait)
                strategy['sgRNAs'] = self.select_sgRNAs_based_on_scores(x, strategy['sgRNAs'])
                evaluation = self.evaluate_strategy(strategy)
                # Objective: Maximize average efficiency and minimize max off-target
                return -evaluation['evaluation_metrics']['average_efficiency'] + evaluation['evaluation_metrics']['max_off_target']
            
            result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=100, seed=42)
            optimized_strategy = self.generate_strategy(target_gene, desired_trait)
            optimized_strategy['sgRNAs'] = self.select_sgRNAs_based_on_scores(result.x, optimized_strategy['sgRNAs'])
            optimized_evaluation = self.evaluate_strategy(optimized_strategy)
            optimized_strategy.update(optimized_evaluation)

            logger.info("CRISPR strategy optimized successfully for gene: %s", target_gene)
            return optimized_strategy
        except Exception as e:
            logger.error("Error optimizing CRISPR strategy: %s", e)
            raise e

    def fetch_gene_sequence(self, gene: str) -> Optional[str]:
        """
        Fetches the gene sequence from the database or external API.
        """
        logger.debug("Fetching gene sequence for gene: %s", gene)
        try:
            query = f"SELECT sequence FROM genes WHERE gene_symbol = '{gene}'"
            result = self.db_connector.execute_query(query)
            if result:
                return result[0]['sequence']
            return None
        except Exception as e:
            logger.error("Error fetching gene sequence: %s", e)
            raise e

    def identify_sgRNAs(self, gene_sequence: str) -> List[Dict[str, Any]]:
        """
        Identifies potential sgRNA targets within the gene sequence.
        """
        logger.debug("Identifying sgRNAs within the gene sequence")
        try:
            sgRNAs = []
            for i in range(len(gene_sequence) - 23):  # 20 nt sgRNA + PAM
                sgRNA = gene_sequence[i:i+20]
                pam = gene_sequence[i+20:i+23]
                if pam.upper() in ['NGG', 'AGG']:
                    sgRNAs.append({
                        'sequence': sgRNA,
                        'position': i+1,
                        'pam': pam
                    })
            logger.info("Identified %d potential sgRNAs", len(sgRNAs))
            return sgRNAs
        except Exception as e:
            logger.error("Error identifying sgRNAs: %s", e)
            raise e

    def select_sgRNAs_based_on_scores(self, scores: np.ndarray, sgRNAs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Selects sgRNAs based on the provided efficiency scores.
        """
        logger.debug("Selecting sgRNAs based on efficiency scores")
        try:
            scored_sgRNAs = []
            for score, sgRNA in zip(scores, sgRNAs):
                sgRNA['efficiency_score'] = score
                scored_sgRNAs.append(sgRNA)
            # Sort sgRNAs by efficiency score descending
            scored_sgRNAs.sort(key=lambda x: x['efficiency_score'], reverse=True)
            selected_sgRNAs = scored_sgRNAs[:self.strategy_params.get('top_sgRNAs', 5)]
            logger.info("Selected top %d sgRNAs based on efficiency", len(selected_sgRNAs))
            return selected_sgRNAs
        except Exception as e:
            logger.error("Error selecting sgRNAs based on scores: %s", e)
            raise e


class GenomeStrategyManager:
    """
    Manages the lifecycle of genome editing strategies within the WAVE ecosystem,
    including generation, evaluation, optimization, and storage.
    """

    def __init__(self, config_path: str = 'config/genome_strategies_config.json'):
        logger.info("Initializing GenomeStrategyManager with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.strategies: List[GenomeStrategy] = []
        self.load_strategies()
        logger.info("GenomeStrategyManager initialized successfully")

    def load_strategies(self):
        """
        Loads available genome editing strategy implementations.
        """
        logger.info("Loading genome editing strategies")
        try:
            # Instantiate strategy classes based on configuration
            strategy_types = self.config.get('strategies', [])
            for strategy_type in strategy_types:
                if strategy_type == 'CRISPR':
                    strategy = CRISPRStrategy()
                # Future strategy types can be added here
                else:
                    logger.warning("Unsupported strategy type: %s", strategy_type)
                    continue
                self.strategies.append(strategy)
                logger.debug("Loaded strategy type: %s", strategy_type)
            logger.info("Loaded %d genome editing strategies", len(self.strategies))
        except Exception as e:
            logger.error("Error loading genome editing strategies: %s", e)
            raise e

    def generate_and_optimize_strategies(self, target_genes: List[str], desired_traits: List[str]):
        """
        Generates and optimizes genome editing strategies for the specified target genes and desired traits.
        """
        logger.info("Generating and optimizing genome editing strategies")
        try:
            for gene, trait in zip(target_genes, desired_traits):
                for strategy in self.strategies:
                    strategy_dict = strategy.generate_strategy(gene, trait)
                    evaluation = strategy.evaluate_strategy(strategy_dict)
                    optimized_strategy = strategy.optimize_strategy(gene, trait)
                    # Save both original and optimized strategies
                    strategy_filename = f"{strategy_dict['strategy_type']}_{gene}_{trait}.json"
                    optimized_filename = f"{strategy_dict['strategy_type']}_{gene}_{trait}_optimized.json"
                    strategy.save_strategy(strategy_dict, strategy_filename)
                    strategy.save_strategy(optimized_strategy, optimized_filename)
            logger.info("Strategy generation and optimization completed successfully")
        except Exception as e:
            logger.error("Error during strategy generation and optimization: %s", e)
            raise e

    def retrieve_strategies(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Retrieves genome editing strategies from the database based on provided filters.
        """
        logger.info("Retrieving genome editing strategies from the database")
        try:
            query = "SELECT * FROM genome_strategies"
            if filters:
                conditions = []
                values = []
                for key, value in filters.items():
                    conditions.append(f"{key} = %s")
                    values.append(value)
                condition_str = " AND ".join(conditions)
                query += f" WHERE {condition_str}"
            data = self.db_connector.execute_query(query, tuple(values) if filters else ())
            df_strategies = pd.DataFrame(data)
            logger.info("Retrieved %d genome editing strategies", len(df_strategies))
            return df_strategies
        except SQLAlchemyError as e:
            logger.error("Database error while retrieving strategies: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error while retrieving strategies: %s", e)
            raise e

    def analyze_strategies(self, df_strategies: pd.DataFrame):
        """
        Analyzes the effectiveness and safety of the retrieved strategies.
        Generates reports and visualizations.
        """
        logger.info("Analyzing retrieved genome editing strategies")
        try:
            # Example analysis: distribution of efficiency scores
            plt.figure(figsize=(10,6))
            sns.histplot(df_strategies['evaluation_metrics'].apply(lambda x: x.get('average_efficiency', 0)), bins=20, kde=True)
            plt.title('Distribution of Average Efficiency Scores')
            plt.xlabel('Average Efficiency')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plot_path = os.path.join(self.config.get('report_output', 'reports/genome_strategies/'), 'efficiency_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info("Efficiency distribution plot saved at %s", plot_path)

            # Example analysis: correlation between efficiency and off-target effects
            df_strategies['average_efficiency'] = df_strategies['evaluation_metrics'].apply(lambda x: x.get('average_efficiency', 0))
            df_strategies['max_off_target'] = df_strategies['evaluation_metrics'].apply(lambda x: x.get('max_off_target', 0))
            plt.figure(figsize=(10,6))
            sns.scatterplot(data=df_strategies, x='average_efficiency', y='max_off_target', hue='strategy_type')
            plt.title('Efficiency vs. Max Off-Target Effects')
            plt.xlabel('Average Efficiency')
            plt.ylabel('Max Off-Target Effects')
            plt.legend(title='Strategy Type')
            plt.tight_layout()
            plot_path = os.path.join(self.config.get('report_output', 'reports/genome_strategies/'), 'efficiency_vs_offtarget.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info("Efficiency vs. Off-Target effects plot saved at %s", plot_path)
        except Exception as e:
            logger.error("Error during strategy analysis: %s", e)
            raise e

    def run_pipeline(self, target_genes: List[str], desired_traits: List[str]):
        """
        Runs the full genome editing strategy pipeline: generation, optimization, retrieval, and analysis.
        """
        logger.info("Running full genome editing strategy pipeline")
        try:
            self.generate_and_optimize_strategies(target_genes, desired_traits)
            df_strategies = self.retrieve_strategies()
            self.analyze_strategies(df_strategies)
            logger.info("Genome editing strategy pipeline completed successfully")
        except Exception as e:
            logger.error("Genome editing strategy pipeline failed: %s", e)
            raise e


def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Genome Strategies Module')
    parser.add_argument('--config', type=str, default='config/genome_strategies_config.json',
                        help='Path to genome strategies configuration file')
    parser.add_argument('--genes', type=str, nargs='+', required=True,
                        help='List of target genes for strategy generation')
    parser.add_argument('--traits', type=str, nargs='+', required=True,
                        help='List of desired traits corresponding to target genes')
    parser.add_argument('--run_pipeline', action='store_true', help='Run the full genome strategies pipeline')
    args = parser.parse_args()

    if len(args.genes) != len(args.traits):
        logger.error("The number of genes and traits must be equal.")
        sys.exit(1)

    try:
        manager = GenomeStrategyManager(config_path=args.config)
        if args.run_pipeline:
            manager.run_pipeline(args.genes, args.traits)
        else:
            logger.error("No action specified. Use --run_pipeline to execute the strategy pipeline.")
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error("Genome strategies operation failed: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()

