import os
import json
import logging
import sys
import time
from typing import Any, Dict, Optional, List

from web3 import Web3
from web3.middleware import geth_poa_middleware
from dotenv import load_dotenv
from eth_account import Account
from eth_utils import to_checksum_address

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionManager:
    def __init__(self, config_path: str = 'config/transaction_manager_config.json'):
        logger.info("Initializing TransactionManager with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.web3 = self.connect_to_blockchain()
        self.account = self.load_account()
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.transaction_timeout = self.config.get('transaction_timeout', 120)  # seconds
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 5)  # seconds
        logger.info("TransactionManager initialized successfully")

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

    def load_account(self) -> Account:
        load_dotenv()
        private_key = os.getenv('TRANSACTION_MANAGER_PRIVATE_KEY')
        if not private_key:
            logger.error("Transaction Manager private key not found in environment variables")
            raise EnvironmentError("Transaction Manager private key not found in environment variables")
        account = self.web3.eth.account.from_key(private_key)
        logger.info("Loaded Transaction Manager account: %s", account.address)
        return account

    def build_transaction(self, to: str, data: bytes, value: int = 0, gas: Optional[int] = None, gas_price: Optional[int] = None) -> Dict[str, Any]:
        logger.debug("Building transaction to %s with value %d and data size %d", to, value, len(data))
        transaction = {
            'to': to_checksum_address(to),
            'value': value,
            'data': data,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'chainId': self.config.get('blockchain', {}).get('chain_id'),
            'gas': gas or self.web3.eth.estimate_gas({
                'to': to_checksum_address(to),
                'from': self.account.address,
                'value': value,
                'data': data
            }),
            'gasPrice': gas_price or self.web3.toWei(self.config.get('gas_price', '20'), 'gwei')
        }
        logger.debug("Transaction built: %s", transaction)
        return transaction

    def sign_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Signing transaction")
        signed_txn = self.account.sign_transaction(transaction)
        logger.debug("Transaction signed: %s", signed_txn)
        return signed_txn

    def send_transaction(self, signed_txn: Dict[str, Any]) -> str:
        logger.debug("Sending signed transaction")
        tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
        logger.info("Transaction sent. Hash: %s", tx_hash.hex())
        return tx_hash.hex()

    def wait_for_receipt(self, tx_hash: str) -> Dict[str, Any]:
        logger.debug("Waiting for transaction receipt for hash: %s", tx_hash)
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.transaction_timeout)
            logger.info("Transaction receipt received for hash: %s", tx_hash)
            return receipt
        except Exception as e:
            logger.error("Error waiting for transaction receipt: %s", e)
            raise e

    def retry_transaction(self, build_tx_func, *args, **kwargs) -> Optional[str]:
        logger.debug("Attempting to send transaction with retry mechanism")
        for attempt in range(1, self.retry_attempts + 1):
            try:
                transaction = build_tx_func(*args, **kwargs)
                signed_txn = self.sign_transaction(transaction)
                tx_hash = self.send_transaction(signed_txn)
                receipt = self.wait_for_receipt(tx_hash)
                if receipt.status == 1:
                    logger.info("Transaction successful on attempt %d: %s", attempt, tx_hash)
                    return tx_hash
                else:
                    logger.warning("Transaction failed on attempt %d: %s", attempt, tx_hash)
            except Exception as e:
                logger.error("Transaction attempt %d failed: %s", attempt, e)
            if attempt < self.retry_attempts:
                logger.info("Retrying transaction after %d seconds...", self.retry_delay)
                time.sleep(self.retry_delay)
        logger.error("All transaction attempts failed")
        return None

    def execute_contract_function(self, contract_function, *args, **kwargs) -> Optional[str]:
        logger.debug("Executing contract function: %s with args: %s and kwargs: %s", contract_function.fn_name, args, kwargs)
        build_tx_func = lambda: contract_function(*args, **kwargs).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'gas': kwargs.get('gas', self.web3.eth.estimate_gas({
                'from': self.account.address,
                'to': contract_function.address,
                'data': contract_function.encodeABI(*args, **kwargs)
            })),
            'gasPrice': self.web3.toWei(self.config.get('gas_price', '20'), 'gwei'),
            'chainId': self.config.get('blockchain', {}).get('chain_id')
        })
        tx_hash = self.retry_transaction(build_tx_func)
        if tx_hash:
            logger.info("Contract function %s executed successfully. Tx Hash: %s", contract_function.fn_name, tx_hash)
        else:
            logger.error("Failed to execute contract function %s", contract_function.fn_name)
        return tx_hash

    def send_data_transaction(self, to: str, data_payload: Dict[str, Any]) -> Optional[str]:
        logger.debug("Preparing to send data transaction to %s with payload: %s", to, data_payload)
        data_json = json.dumps(data_payload)
        data_bytes = self.web3.toBytes(text=data_json)
        tx_hash = self.retry_transaction(self.build_transaction, to=to, data=data_bytes, value=0)
        if tx_hash:
            logger.info("Data transaction sent successfully. Tx Hash: %s", tx_hash)
        else:
            logger.error("Failed to send data transaction to %s", to)
        return tx_hash

    def approve_token_spend(self, token_contract, spender: str, amount: int) -> Optional[str]:
        logger.debug("Approving token spend for spender: %s with amount: %d", spender, amount)
        approve_function = token_contract.functions.approve(spender, amount)
        tx_hash = self.execute_contract_function(approve_function)
        return tx_hash

    def transfer_tokens(self, token_contract, to: str, amount: int) -> Optional[str]:
        logger.debug("Transferring tokens to %s with amount: %d", to, amount)
        transfer_function = token_contract.functions.transfer(to, amount)
        tx_hash = self.execute_contract_function(transfer_function)
        return tx_hash

    def get_transaction_status(self, tx_hash: str) -> Optional[bool]:
        logger.debug("Checking status for transaction hash: %s", tx_hash)
        try:
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            status = receipt.status == 1
            logger.info("Transaction %s status: %s", tx_hash, status)
            return status
        except Exception as e:
            logger.error("Error fetching transaction receipt: %s", e)
            return None

    def log_transaction(self, tx_hash: str, description: str, status: bool):
        logger.debug("Logging transaction %s with description: %s and status: %s", tx_hash, description, status)
        log_entry = {
            'tx_hash': tx_hash,
            'description': description,
            'status': status,
            'timestamp': int(time.time())
        }
        self.db_connector.insert_record('transactions', log_entry)
        logger.info("Transaction %s logged successfully", tx_hash)

    def run_transaction(self, to: str, data_payload: Dict[str, Any], description: str):
        logger.info("Running transaction to %s with description: %s", to, description)
        tx_hash = self.send_data_transaction(to, data_payload)
        if tx_hash:
            status = self.get_transaction_status(tx_hash)
            self.log_transaction(tx_hash, description, status)
            return tx_hash, status
        else:
            logger.error("Failed to run transaction to %s", to)
            return None, False

    def batch_run_transactions(self, transactions: List[Dict[str, Any]]):
        logger.info("Starting batch transaction processing for %d transactions", len(transactions))
        for tx in transactions:
            to = tx.get('to')
            data_payload = tx.get('data_payload', {})
            description = tx.get('description', 'No description provided')
            self.run_transaction(to, data_payload, description)
        logger.info("Batch transaction processing completed successfully")

    def schedule_transaction(self, to: str, data_payload: Dict[str, Any], description: str, delay: int):
        logger.info("Scheduling transaction to %s with delay of %d seconds", to, delay)
        time.sleep(delay)
        self.run_transaction(to, data_payload, description)

    def monitor_pending_transactions(self):
        logger.info("Monitoring pending transactions")
        while True:
            pending = self.web3.eth.filter('pending').get_new_entries()
            for tx_hash in pending:
                try:
                    tx = self.web3.eth.get_transaction(tx_hash)
                    logger.debug("Pending transaction detected: %s", tx_hash)
                    # Implement custom logic for pending transactions
                except Exception as e:
                    logger.error("Error fetching pending transaction %s: %s", tx_hash, e)
            time.sleep(10)  # Polling interval

    def generate_transaction_report(self, start_time: int, end_time: int, output_path: str = 'reports/transaction_report.json'):
        logger.debug("Generating transaction report from %d to %d", start_time, end_time)
        query = """
            SELECT tx_hash, description, status, timestamp
            FROM transactions
            WHERE timestamp BETWEEN %s AND %s
        """
        results = self.db_connector.execute_query(query, (start_time, end_time))
        report = [dict(row) for row in results]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as report_file:
            json.dump(report, report_file, indent=4)
        logger.info("Transaction report generated at %s", output_path)

    def handle_failed_transactions(self):
        logger.info("Handling failed transactions")
        query = """
            SELECT tx_hash, description
            FROM transactions
            WHERE status = FALSE
        """
        failed_transactions = self.db_connector.execute_query(query)
        for tx in failed_transactions:
            tx_hash = tx['tx_hash']
            description = tx['description']
            logger.info("Retrying failed transaction %s: %s", tx_hash, description)
            # Implement logic to retry the transaction based on description or other metadata
            # This could involve re-sending the same data_payload or adjusting parameters
        logger.info("Failed transactions handled successfully")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Transaction Manager Module')
    parser.add_argument('--config', type=str, default='config/transaction_manager_config.json', help='Path to transaction manager configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run single transaction
    run_parser = subparsers.add_parser('run', help='Run a single transaction')
    run_parser.add_argument('--to', type=str, required=True, help='Recipient address')
    run_parser.add_argument('--data', type=str, required=True, help='Data payload in JSON string')
    run_parser.add_argument('--description', type=str, default='No description provided', help='Transaction description')

    # Run batch transactions
    batch_parser = subparsers.add_parser('batch', help='Run batch transactions')
    batch_parser.add_argument('--file', type=str, required=True, help='Path to batch transactions JSON file')

    # Schedule transaction
    schedule_parser = subparsers.add_parser('schedule', help='Schedule a transaction with delay')
    schedule_parser.add_argument('--to', type=str, required=True, help='Recipient address')
    schedule_parser.add_argument('--data', type=str, required=True, help='Data payload in JSON string')
    schedule_parser.add_argument('--description', type=str, default='No description provided', help='Transaction description')
    schedule_parser.add_argument('--delay', type=int, required=True, help='Delay in seconds before executing the transaction')

    # Generate report
    report_parser = subparsers.add_parser('report', help='Generate transaction report')
    report_parser.add_argument('--start', type=int, required=True, help='Start timestamp')
    report_parser.add_argument('--end', type=int, required=True, help='End timestamp')
    report_parser.add_argument('--output', type=str, default='reports/transaction_report.json', help='Output path for the report')

    # Handle failed transactions
    handle_parser = subparsers.add_parser('handle_failed', help='Handle failed transactions')

    args = parser.parse_args()

    manager = TransactionManager(config_path=args.config)

    if args.command == 'run':
        try:
            data_payload = json.loads(args.data)
            tx_hash, status = manager.run_transaction(args.to, data_payload, args.description)
            if tx_hash:
                logger.info("Transaction executed with hash: %s and status: %s", tx_hash, status)
            else:
                logger.error("Transaction execution failed")
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for data payload")
            sys.exit(1)
        except Exception as e:
            logger.error("Error running transaction: %s", e)
            sys.exit(1)

    elif args.command == 'batch':
        try:
            if not os.path.exists(args.file):
                logger.error("Batch transactions file not found: %s", args.file)
                sys.exit(1)
            with open(args.file, 'r') as f:
                transactions = json.load(f)
            manager.batch_run_transactions(transactions)
            logger.info("Batch transactions executed successfully")
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in batch transactions file")
            sys.exit(1)
        except Exception as e:
            logger.error("Error running batch transactions: %s", e)
            sys.exit(1)

    elif args.command == 'schedule':
        try:
            data_payload = json.loads(args.data)
            manager.schedule_transaction(args.to, data_payload, args.description, args.delay)
            logger.info("Scheduled transaction will execute after %d seconds", args.delay)
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for data payload")
            sys.exit(1)
        except Exception as e:
            logger.error("Error scheduling transaction: %s", e)
            sys.exit(1)

    elif args.command == 'report':
        try:
            manager.generate_transaction_report(args.start, args.end, args.output)
            logger.info("Transaction report generated successfully")
        except Exception as e:
            logger.error("Error generating transaction report: %s", e)
            sys.exit(1)

    elif args.command == 'handle_failed':
        try:
            manager.handle_failed_transactions()
            logger.info("Failed transactions handled successfully")
        except Exception as e:
            logger.error("Error handling failed transactions: %s", e)
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()

