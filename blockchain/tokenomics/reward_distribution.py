import os
import json
import logging
import sys
from typing import List, Dict, Any

from web3 import Web3
from web3.middleware import geth_poa_middleware
from dotenv import load_dotenv

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardDistributor:
    def __init__(self, config_path: str = 'config/reward_distribution_config.json'):
        logger.info("Initializing RewardDistributor with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.web3 = self.connect_to_blockchain()
        self.contract = self.load_contract()
        self.account = self.load_account()
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.reward_token_address = self.config.get('reward_token_address')
        self.reward_amount = self.config.get('reward_amount')
        logger.info("RewardDistributor initialized successfully")

    def connect_to_blockchain(self) -> Web3:
        logger.debug("Connecting to blockchain network")
        network = self.config.get('blockchain', {})
        rpc_url = network.get('rpc_url')
        chain_id = network.get('chain_id')
        if not rpc_url:
            logger.error("RPC URL not specified in configuration")
            raise ValueError("RPC URL not specified in configuration")
        
        web3 = Web3(Web3.HTTPProvider(rpc_url))
        if chain_id == 1337:  # Example for local Ganache
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        if not web3.isConnected():
            logger.error("Failed to connect to the blockchain network at %s", rpc_url)
            raise ConnectionError(f"Failed to connect to the blockchain network at {rpc_url}")
        logger.info("Connected to blockchain network: %s", rpc_url)
        return web3

    def load_account(self) -> Any:
        load_dotenv()
        private_key = os.getenv('REWARD_DISTRIBUTOR_PRIVATE_KEY')
        if not private_key:
            logger.error("Reward distributor private key not found in environment variables")
            raise EnvironmentError("Reward distributor private key not found in environment variables")
        account = self.web3.eth.account.from_key(private_key)
        logger.info("Loaded reward distributor account: %s", account.address)
        return account

    def load_contract(self) -> Any:
        logger.debug("Loading reward smart contract")
        contract_address = self.config.get('reward_contract_address')
        abi_path = self.config.get('reward_contract_abi_path')
        if not contract_address or not abi_path:
            logger.error("Reward contract address or ABI path not specified in configuration")
            raise ValueError("Reward contract address or ABI path not specified in configuration")
        if not os.path.exists(abi_path):
            logger.error("ABI file not found at path: %s", abi_path)
            raise FileNotFoundError(f"ABI file not found at path: {abi_path}")
        with open(abi_path, 'r') as abi_file:
            contract_abi = json.load(abi_file)
        contract = self.web3.eth.contract(address=contract_address, abi=contract_abi)
        logger.info("Reward smart contract loaded at address: %s", contract_address)
        return contract

    def get_eligible_participants(self) -> List[Dict[str, Any]]:
        logger.debug("Fetching eligible participants for rewards")
        query = "SELECT user_address, contribution_score FROM user_contributions WHERE rewarded = FALSE AND contribution_score >= %s"
        threshold = self.config.get('contribution_threshold', 100)
        results = self.db_connector.execute_query(query, (threshold,))
        eligible_participants = [{'address': row['user_address'], 'score': row['contribution_score']} for row in results]
        logger.info("Found %d eligible participants for rewards", len(eligible_participants))
        return eligible_participants

    def distribute_rewards(self, participants: List[Dict[str, Any]]):
        logger.info("Starting reward distribution to %d participants", len(participants))
        for participant in participants:
            address = participant['address']
            score = participant['score']
            try:
                txn = self.contract.functions.transfer(address, self.reward_amount).build_transaction({
                    'from': self.account.address,
                    'nonce': self.web3.eth.get_transaction_count(self.account.address),
                    'gas': self.config.get('gas', 100000),
                    'gasPrice': self.web3.toWei(self.config.get('gas_price', '20'), 'gwei')
                })
                signed_txn = self.web3.eth.account.sign_transaction(txn, private_key=self.account.key)
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                logger.info("Sent reward transaction to %s. Tx Hash: %s", address, tx_hash.hex())
                tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                if tx_receipt.status == 1:
                    logger.info("Reward distributed successfully to %s", address)
                    self.mark_as_rewarded(address)
                else:
                    logger.error("Reward transaction failed for %s. Receipt: %s", address, tx_receipt)
            except Exception as e:
                logger.error("Error distributing reward to %s: %s", address, e)

    def mark_as_rewarded(self, address: str):
        logger.debug("Marking address %s as rewarded", address)
        update_query = "UPDATE user_contributions SET rewarded = TRUE WHERE user_address = %s"
        self.db_connector.execute_query(update_query, (address,))
        logger.info("Address %s marked as rewarded in the database", address)

    def generate_report(self, participants: List[Dict[str, Any]]):
        logger.debug("Generating reward distribution report")
        report = {
            'total_rewards': len(participants) * self.reward_amount,
            'participants': participants
        }
        report_path = self.config.get('report_path', 'reports/reward_distribution_report.json')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as report_file:
            json.dump(report, report_file, indent=4)
        logger.info("Reward distribution report generated at %s", report_path)

    def run_distribution_cycle(self):
        logger.info("Running reward distribution cycle")
        participants = self.get_eligible_participants()
        if not participants:
            logger.info("No eligible participants found for rewards")
            return
        self.distribute_rewards(participants)
        self.generate_report(participants)
        logger.info("Reward distribution cycle completed successfully")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Reward Distribution Module')
    parser.add_argument('--config', type=str, default='config/reward_distribution_config.json', help='Path to reward distribution configuration file')
    args = parser.parse_args()

    try:
        distributor = RewardDistributor(config_path=args.config)
        distributor.run_distribution_cycle()
    except Exception as e:
        logger.error("Reward distribution failed: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()

