import os
import json
import logging
import sys
from typing import Any, Dict, Optional

from web3 import Web3
from web3.middleware import geth_poa_middleware
from dotenv import load_dotenv
import ipfshttpclient

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecentralizedStorage:
    def __init__(self, config_path: str = 'config/decentralized_storage_config.json'):
        logger.info("Initializing DecentralizedStorage with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.web3 = self.connect_to_blockchain()
        self.contract = self.load_contract()
        self.ipfs_client = self.connect_to_ipfs()
        self.account = self.load_account()
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        logger.info("DecentralizedStorage initialized successfully")

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
        private_key = os.getenv('DEPLOYER_PRIVATE_KEY')
        if not private_key:
            logger.error("Deployer private key not found in environment variables")
            raise EnvironmentError("Deployer private key not found in environment variables")
        account = self.web3.eth.account.from_key(private_key)
        logger.info("Loaded deployer account: %s", account.address)
        return account

    def load_contract(self) -> Any:
        logger.debug("Loading smart contract")
        contract_address = self.config.get('contract_address')
        abi_path = self.config.get('contract_abi_path')
        if not contract_address or not abi_path:
            logger.error("Contract address or ABI path not specified in configuration")
            raise ValueError("Contract address or ABI path not specified in configuration")
        if not os.path.exists(abi_path):
            logger.error("ABI file not found at path: %s", abi_path)
            raise FileNotFoundError(f"ABI file not found at path: {abi_path}")
        with open(abi_path, 'r') as abi_file:
            contract_abi = json.load(abi_file)
        contract = self.web3.eth.contract(address=contract_address, abi=contract_abi)
        logger.info("Smart contract loaded at address: %s", contract_address)
        return contract

    def connect_to_ipfs(self) -> ipfshttpclient.Client:
        logger.debug("Connecting to IPFS node")
        ipfs_config = self.config.get('ipfs', {})
        host = ipfs_config.get('host', '127.0.0.1')
        port = ipfs_config.get('port', 5001)
        try:
            client = ipfshttpclient.connect(f'/dns/{host}/tcp/{port}/http')
            logger.info("Connected to IPFS node at /dns/%s/tcp/%s/http", host, port)
            return client
        except Exception as e:
            logger.error("Failed to connect to IPFS node: %s", e)
            raise ConnectionError(f"Failed to connect to IPFS node: {e}")

    def upload_to_ipfs(self, file_path: str) -> str:
        logger.debug("Uploading file to IPFS: %s", file_path)
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            res = self.ipfs_client.add(file_path)
            ipfs_hash = res['Hash']
            logger.info("File uploaded to IPFS with hash: %s", ipfs_hash)
            return ipfs_hash
        except Exception as e:
            logger.error("Failed to upload file to IPFS: %s", e)
            raise e

    def store_data_on_blockchain(self, data_hash: str, metadata: str) -> str:
        logger.debug("Storing data hash on blockchain")
        try:
            txn = self.contract.functions.storeData(data_hash, metadata).build_transaction({
                'from': self.account.address,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'gas': self.config.get('gas', 300000),
                'gasPrice': self.web3.toWei(self.config.get('gas_price', '20'), 'gwei')
            })
            signed_txn = self.web3.eth.account.sign_transaction(txn, private_key=self.account.key)
            logger.info("Sending transaction to store data on blockchain")
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            logger.info("Transaction sent. Hash: %s", tx_hash.hex())
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            logger.info("Transaction receipt received. Status: %s", tx_receipt.status)
            if tx_receipt.status == 1:
                logger.info("Data hash stored successfully on blockchain")
                return tx_hash.hex()
            else:
                logger.error("Transaction failed with receipt: %s", tx_receipt)
                raise Exception("Failed to store data on blockchain")
        except Exception as e:
            logger.error("Error storing data on blockchain: %s", e)
            raise e

    def retrieve_data_from_blockchain(self, record_id: int) -> Dict[str, Any]:
        logger.debug("Retrieving data from blockchain for record ID: %d", record_id)
        try:
            data_record = self.contract.functions.getData(record_id).call()
            logger.info("Data retrieved from blockchain: %s", data_record)
            return {
                "id": data_record[0],
                "dataHash": data_record[1],
                "timestamp": data_record[2],
                "uploader": data_record[3],
                "metadata": data_record[4]
            }
        except Exception as e:
            logger.error("Error retrieving data from blockchain: %s", e)
            raise e

    def download_from_ipfs(self, ipfs_hash: str, destination_path: str):
        logger.debug("Downloading file from IPFS with hash: %s", ipfs_hash)
        try:
            self.ipfs_client.get(ipfs_hash, target=destination_path)
            logger.info("File downloaded from IPFS to: %s", destination_path)
        except Exception as e:
            logger.error("Failed to download file from IPFS: %s", e)
            raise e

    def link_ipfs_to_database(self, ipfs_hash: str, metadata: Dict[str, Any]):
        logger.debug("Linking IPFS hash to database with metadata: %s", metadata)
        try:
            record = {
                "ipfs_hash": ipfs_hash,
                "metadata": metadata
            }
            self.db_connector.insert_record("decentralized_storage", record)
            logger.info("IPFS hash linked to database successfully")
        except Exception as e:
            logger.error("Failed to link IPFS hash to database: %s", e)
            raise e

    def run_storage_pipeline(self, file_path: str, metadata: str):
        logger.debug("Running storage pipeline for file: %s", file_path)
        try:
            # Upload file to IPFS
            ipfs_hash = self.upload_to_ipfs(file_path)
            
            # Store IPFS hash and metadata on blockchain
            tx_hash = self.store_data_on_blockchain(ipfs_hash, metadata)
            
            # Link IPFS hash to database
            metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            self.link_ipfs_to_database(ipfs_hash, metadata_dict)
            
            logger.info("Storage pipeline completed successfully. Transaction Hash: %s", tx_hash)
            return tx_hash
        except Exception as e:
            logger.error("Storage pipeline failed: %s", e)
            raise e

    def run_retrieval_pipeline(self, record_id: int, download_path: str):
        logger.debug("Running retrieval pipeline for record ID: %d", record_id)
        try:
            # Retrieve data record from blockchain
            data_record = self.retrieve_data_from_blockchain(record_id)
            ipfs_hash = data_record['dataHash']
            metadata = data_record['metadata']
            
            # Download file from IPFS
            self.download_from_ipfs(ipfs_hash, download_path)
            
            # Optionally, process metadata or perform additional actions
            logger.info("Retrieval pipeline completed successfully for record ID: %d", record_id)
            return {
                "data_record": data_record,
                "download_path": download_path
            }
        except Exception as e:
            logger.error("Retrieval pipeline failed: %s", e)
            raise e

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Decentralized Storage Module')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload a file to IPFS and store its hash on the blockchain')
    upload_parser.add_argument('--file', type=str, required=True, help='Path to the file to upload')
    upload_parser.add_argument('--metadata', type=str, required=True, help='Metadata JSON string')

    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve a file from IPFS using the record ID from the blockchain')
    retrieve_parser.add_argument('--record_id', type=int, required=True, help='Record ID to retrieve')
    retrieve_parser.add_argument('--download_path', type=str, required=True, help='Path to download the file')

    args = parser.parse_args()

    storage = DecentralizedStorage()

    if args.command == 'upload':
        try:
            tx_hash = storage.run_storage_pipeline(args.file, args.metadata)
            logger.info("Upload successful. Transaction Hash: %s", tx_hash)
        except Exception as e:
            logger.error("Upload failed: %s", e)
            sys.exit(1)
    elif args.command == 'retrieve':
        try:
            retrieval_info = storage.run_retrieval_pipeline(args.record_id, args.download_path)
            logger.info("Retrieve successful. Data: %s", retrieval_info)
        except Exception as e:
            logger.error("Retrieve failed: %s", e)
            sys.exit(1)
    else:
        parser.print_help()

