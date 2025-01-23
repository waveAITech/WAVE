import os
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from web3 import Web3
from web3.middleware import geth_poa_middleware
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainAnalyzer:
    """
    Analyzes blockchain data related to the WAVE ecosystem, including transactions,
    smart contract interactions, and event monitoring.
    """

    def __init__(self, config_path: str = 'config/blockchain_analysis_config.json'):
        logger.info("Initializing BlockchainAnalyzer with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.web3 = self.connect_to_blockchain()
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.contracts = self.load_contracts()
        logger.info("BlockchainAnalyzer initialized successfully")

    def connect_to_blockchain(self) -> Web3:
        logger.debug("Connecting to blockchain network")
        network = self.config.get('blockchain', {})
        rpc_url = network.get('rpc_url')
        chain_id = network.get('chain_id')
        if not rpc_url:
            logger.error("RPC URL not specified in configuration")
            raise ValueError("RPC URL not specified in configuration")
        
        web3 = Web3(Web3.HTTPProvider(rpc_url))
        if chain_id in [1337, 5777]:  # Example for local networks like Ganache
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        if not web3.isConnected():
            logger.error("Failed to connect to the blockchain network at %s", rpc_url)
            raise ConnectionError(f"Failed to connect to the blockchain network at {rpc_url}")
        logger.info("Connected to blockchain network: %s", rpc_url)
        return web3

    def load_contracts(self) -> Dict[str, Any]:
        logger.debug("Loading smart contracts for analysis")
        contracts_config = self.config.get('contracts', [])
        contracts = {}
        for contract in contracts_config:
            name = contract.get('name')
            address = contract.get('address')
            abi_path = contract.get('abi_path')
            if not all([name, address, abi_path]):
                logger.warning("Incomplete contract configuration: %s", contract)
                continue
            if not os.path.exists(abi_path):
                logger.error("ABI file not found at path: %s", abi_path)
                continue
            with open(abi_path, 'r') as abi_file:
                contract_abi = json.load(abi_file)
            contract_instance = self.web3.eth.contract(address=self.web3.toChecksumAddress(address), abi=contract_abi)
            contracts[name] = contract_instance
            logger.info("Loaded contract '%s' at address %s", name, address)
        return contracts

    def fetch_transactions(self, start_block: int, end_block: int) -> pd.DataFrame:
        logger.info("Fetching transactions from block %d to %d", start_block, end_block)
        transactions = []
        for block_num in range(start_block, end_block + 1):
            try:
                block = self.web3.eth.get_block(block_num, full_transactions=True)
                for tx in block.transactions:
                    tx_data = {
                        'hash': tx.hash.hex(),
                        'from': tx['from'],
                        'to': tx.to,
                        'value': self.web3.fromWei(tx.value, 'ether'),
                        'gas': tx.gas,
                        'gasPrice': self.web3.fromWei(tx.gasPrice, 'gwei'),
                        'blockNumber': tx.blockNumber,
                        'timestamp': block.timestamp
                    }
                    transactions.append(tx_data)
                logger.debug("Fetched block %d with %d transactions", block_num, len(block.transactions))
            except Exception as e:
                logger.error("Error fetching block %d: %s", block_num, e)
        df_transactions = pd.DataFrame(transactions)
        logger.info("Fetched a total of %d transactions", len(df_transactions))
        return df_transactions

    def analyze_transaction_volume(self, df_transactions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Analyzing transaction volume")
        df_volume = df_transactions.groupby('blockNumber').size().reset_index(name='transaction_count')
        logger.debug("Transaction volume analysis completed")
        return df_volume

    def analyze_gas_usage(self, df_transactions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Analyzing gas usage")
        df_gas = df_transactions.groupby('blockNumber')['gasPrice'].mean().reset_index(name='average_gas_price_gwei')
        logger.debug("Gas usage analysis completed")
        return df_gas

    def analyze_contract_interactions(self, df_transactions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Analyzing smart contract interactions")
        contract_addresses = [contract.address for contract in self.contracts.values()]
        df_contract = df_transactions[df_transactions['to'].isin(contract_addresses)]
        interaction_counts = df_contract.groupby('to').size().reset_index(name='interaction_count')
        interaction_counts['contract_name'] = interaction_counts['to'].apply(lambda addr: self.get_contract_name(addr))
        logger.debug("Contract interactions analysis completed")
        return interaction_counts

    def get_contract_name(self, address: str) -> str:
        for name, contract in self.contracts.items():
            if contract.address.lower() == address.lower():
                return name
        return "Unknown"

    def detect_anomalies(self, df_transactions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Detecting anomalies in transactions")
        # Simple anomaly detection based on gas price threshold
        gas_price_threshold = self.config.get('anomaly_detection', {}).get('gas_price_threshold', 100)
        anomalies = df_transactions[df_transactions['gasPrice'] > gas_price_threshold]
        logger.info("Detected %d anomalous transactions based on gas price", len(anomalies))
        return anomalies

    def generate_reports(self, df_volume: pd.DataFrame, df_gas: pd.DataFrame, df_contract: pd.DataFrame, df_anomalies: pd.DataFrame):
        logger.info("Generating analysis reports")
        report_dir = self.config.get('report_output', 'reports/blockchain_analysis/')
        os.makedirs(report_dir, exist_ok=True)

        # Transaction Volume Report
        volume_path = os.path.join(report_dir, 'transaction_volume.csv')
        df_volume.to_csv(volume_path, index=False)
        logger.info("Transaction volume report saved at %s", volume_path)

        # Gas Usage Report
        gas_path = os.path.join(report_dir, 'gas_usage.csv')
        df_gas.to_csv(gas_path, index=False)
        logger.info("Gas usage report saved at %s", gas_path)

        # Contract Interactions Report
        contract_path = os.path.join(report_dir, 'contract_interactions.csv')
        df_contract.to_csv(contract_path, index=False)
        logger.info("Contract interactions report saved at %s", contract_path)

        # Anomalies Report
        anomalies_path = os.path.join(report_dir, 'anomalous_transactions.csv')
        df_anomalies.to_csv(anomalies_path, index=False)
        logger.info("Anomalous transactions report saved at %s", anomalies_path)

        # Visualization
        self.visualize_data(df_volume, df_gas, df_contract)

    def visualize_data(self, df_volume: pd.DataFrame, df_gas: pd.DataFrame, df_contract: pd.DataFrame):
        logger.info("Creating visualizations for analysis reports")
        report_dir = self.config.get('report_output', 'reports/blockchain_analysis/')
        os.makedirs(report_dir, exist_ok=True)

        # Transaction Volume Over Time
        plt.figure(figsize=(10,6))
        plt.plot(df_volume['blockNumber'], df_volume['transaction_count'], label='Transaction Count')
        plt.xlabel('Block Number')
        plt.ylabel('Number of Transactions')
        plt.title('Transaction Volume Over Blocks')
        plt.legend()
        volume_plot = os.path.join(report_dir, 'transaction_volume.png')
        plt.savefig(volume_plot)
        plt.close()
        logger.info("Transaction volume plot saved at %s", volume_plot)

        # Average Gas Price Over Time
        plt.figure(figsize=(10,6))
        plt.plot(df_gas['blockNumber'], df_gas['average_gas_price_gwei'], label='Average Gas Price (Gwei)', color='orange')
        plt.xlabel('Block Number')
        plt.ylabel('Average Gas Price (Gwei)')
        plt.title('Average Gas Price Over Blocks')
        plt.legend()
        gas_plot = os.path.join(report_dir, 'gas_usage.png')
        plt.savefig(gas_plot)
        plt.close()
        logger.info("Gas usage plot saved at %s", gas_plot)

        # Contract Interactions Bar Chart
        plt.figure(figsize=(10,6))
        plt.bar(df_contract['contract_name'], df_contract['interaction_count'], color='green')
        plt.xlabel('Contract Name')
        plt.ylabel('Number of Interactions')
        plt.title('Smart Contract Interactions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        contract_plot = os.path.join(report_dir, 'contract_interactions.png')
        plt.savefig(contract_plot)
        plt.close()
        logger.info("Contract interactions plot saved at %s", contract_plot)

    def run_analysis(self):
        logger.info("Starting blockchain data analysis")
        latest_block = self.web3.eth.block_number
        start_block = self.config.get('analysis', {}).get('start_block', latest_block - 1000)
        end_block = self.config.get('analysis', {}).get('end_block', latest_block)

        df_transactions = self.fetch_transactions(start_block, end_block)
        df_volume = self.analyze_transaction_volume(df_transactions)
        df_gas = self.analyze_gas_usage(df_transactions)
        df_contract = self.analyze_contract_interactions(df_transactions)
        df_anomalies = self.detect_anomalies(df_transactions)

        self.generate_reports(df_volume, df_gas, df_contract, df_anomalies)
        logger.info("Blockchain data analysis completed successfully")

    def save_analysis_to_database(self, df_transactions: pd.DataFrame, df_volume: pd.DataFrame, df_gas: pd.DataFrame, df_contract: pd.DataFrame, df_anomalies: pd.DataFrame):
        logger.info("Saving analysis results to the database")
        try:
            self.db_connector.insert_dataframe('transactions', df_transactions)
            self.db_connector.insert_dataframe('transaction_volume', df_volume)
            self.db_connector.insert_dataframe('gas_usage', df_gas)
            self.db_connector.insert_dataframe('contract_interactions', df_contract)
            self.db_connector.insert_dataframe('anomalous_transactions', df_anomalies)
            logger.info("Analysis results saved to the database successfully")
        except Exception as e:
            logger.error("Error saving analysis results to the database: %s", e)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Blockchain Analysis Module')
    parser.add_argument('--config', type=str, default='config/blockchain_analysis_config.json', help='Path to blockchain analysis configuration file')
    parser.add_argument('--save_db', action='store_true', help='Save analysis results to the database')
    args = parser.parse_args()

    try:
        analyzer = BlockchainAnalyzer(config_path=args.config)
        analyzer.run_analysis()
        if args.save_db:
            # Assuming that the last fetched dataframes are stored as attributes
            analyzer.save_analysis_to_database(
                analyzer.fetch_transactions(
                    analyzer.config.get('analysis', {}).get('start_block', analyzer.web3.eth.block_number - 1000),
                    analyzer.config.get('analysis', {}).get('end_block', analyzer.web3.eth.block_number)
                ),
                analyzer.analyze_transaction_volume(analyzer.fetch_transactions(
                    analyzer.config.get('analysis', {}).get('start_block', analyzer.web3.eth.block_number - 1000),
                    analyzer.config.get('analysis', {}).get('end_block', analyzer.web3.eth.block_number)
                )),
                analyzer.analyze_gas_usage(analyzer.fetch_transactions(
                    analyzer.config.get('analysis', {}).get('start_block', analyzer.web3.eth.block_number - 1000),
                    analyzer.config.get('analysis', {}).get('end_block', analyzer.web3.eth.block_number)
                )),
                analyzer.analyze_contract_interactions(analyzer.fetch_transactions(
                    analyzer.config.get('analysis', {}).get('start_block', analyzer.web3.eth.block_number - 1000),
                    analyzer.config.get('analysis', {}).get('end_block', analyzer.web3.eth.block_number)
                )),
                analyzer.detect_anomalies(analyzer.fetch_transactions(
                    analyzer.config.get('analysis', {}).get('start_block', analyzer.web3.eth.block_number - 1000),
                    analyzer.config.get('analysis', {}).get('end_block', analyzer.web3.eth.block_number)
                ))
            )
    except Exception as e:
        logger.error("Blockchain analysis failed: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()

