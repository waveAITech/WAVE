import os
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

from web3 import Web3
from web3.middleware import geth_poa_middleware
from solcx import compile_source, install_solc
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContractDeployer:
    def __init__(self, config_path: str = 'config/deploy_config.json'):
        logger.info("Initializing ContractDeployer with config path: %s", config_path)
        self.config = self.load_config(config_path)
        self.web3 = self.connect_to_network()
        self.account = self.load_account()
        self.contracts = self.config.get('contracts', [])
        logger.info("ContractDeployer initialized successfully")

    def load_config(self, path: str) -> Dict[str, Any]:
        logger.debug("Loading deployment configuration from: %s", path)
        if not os.path.exists(path):
            logger.error("Configuration file not found: %s", path)
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, 'r') as file:
            config = json.load(file)
        logger.debug("Deployment configuration loaded")
        return config

    def connect_to_network(self) -> Web3:
        network = self.config.get('network', {})
        rpc_url = network.get('rpc_url')
        if not rpc_url:
            logger.error("RPC URL not specified in configuration")
            raise ValueError("RPC URL not specified in configuration")
        web3 = Web3(Web3.HTTPProvider(rpc_url))
        if network.get('chain_id') == 1337:  # Example for local Ganache
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

    def compile_contract(self, contract_path: str) -> Tuple[Dict[str, Any], str]:
        logger.debug("Compiling contract at path: %s", contract_path)
        if not os.path.exists(contract_path):
            logger.error("Contract file not found: %s", contract_path)
            raise FileNotFoundError(f"Contract file not found: {contract_path}")
        with open(contract_path, 'r') as file:
            source = file.read()
        # Install the specified Solidity compiler version if not already installed
        solc_version = self.config.get('solc_version', '0.8.0')
        try:
            install_solc(solc_version)
            logger.debug("Solidity compiler version %s installed", solc_version)
        except Exception as e:
            logger.error("Error installing Solidity compiler: %s", e)
            raise e
        compiled_sol = compile_source(
            source,
            output_values=['abi', 'bin'],
            solc_version=solc_version
        )
        contract_id, contract_interface = compiled_sol.popitem()
        logger.info("Contract compiled successfully: %s", contract_id)
        return contract_interface, contract_id

    def deploy_contract(self, abi: Dict[str, Any], bytecode: str, constructor_args: list = []) -> str:
        logger.debug("Deploying contract with ABI: %s", abi)
        Contract = self.web3.eth.contract(abi=abi, bytecode=bytecode)
        construct_txn = Contract.constructor(*constructor_args).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'gas': self.config.get('gas', 3000000),
            'gasPrice': self.web3.toWei(self.config.get('gas_price', '20'), 'gwei')
        })
        signed_txn = self.web3.eth.account.sign_transaction(construct_txn, private_key=self.account.privateKey)
        logger.info("Sending transaction to deploy contract")
        tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
        logger.info("Transaction sent. Hash: %s", tx_hash.hex())
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = tx_receipt.contractAddress
        logger.info("Contract deployed at address: %s", contract_address)
        return contract_address

    def verify_deployment(self, contract_address: str, expected_abi: Dict[str, Any]) -> bool:
        logger.debug("Verifying contract deployment at address: %s", contract_address)
        contract = self.web3.eth.contract(address=contract_address, abi=expected_abi)
        try:
            # Example verification: check if contract code exists at the address
            code = self.web3.eth.get_code(contract_address)
            if code == b'':
                logger.error("No contract code found at address: %s", contract_address)
                return False
            logger.info("Contract verification successful for address: %s", contract_address)
            return True
        except Exception as e:
            logger.error("Error during contract verification: %s", e)
            return False

    def deploy_all_contracts(self):
        logger.info("Starting deployment of all contracts")
        deployed_contracts = {}
        for contract in self.contracts:
            name = contract.get('name')
            path = contract.get('path')
            constructor_args = contract.get('constructor_args', [])
            logger.info("Deploying contract: %s", name)
            abi, contract_id = self.compile_contract(path)
            bytecode = abi.get('bin') or abi.get('bin-runtime')
            if not bytecode:
                logger.error("Bytecode not found for contract: %s", name)
                continue
            contract_address = self.deploy_contract(abi['abi'], bytecode, constructor_args)
            is_verified = self.verify_deployment(contract_address, abi['abi'])
            deployed_contracts[name] = {
                'address': contract_address,
                'verified': is_verified
            }
            logger.info("Contract %s deployed at %s (Verified: %s)", name, contract_address, is_verified)
        logger.info("All contracts deployed successfully")
        self.save_deployment_info(deployed_contracts)

    def save_deployment_info(self, deployed_contracts: Dict[str, Any]):
        output_path = self.config.get('deployment_output', 'deployment_info.json')
        logger.debug("Saving deployment information to: %s", output_path)
        with open(output_path, 'w') as file:
            json.dump(deployed_contracts, file, indent=4)
        logger.info("Deployment information saved successfully")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Smart Contracts Deployment Script')
    parser.add_argument('--config', type=str, default='config/deploy_config.json', help='Path to deployment configuration file')
    args = parser.parse_args()

    try:
        deployer = ContractDeployer(config_path=args.config)
        deployer.deploy_all_contracts()
    except Exception as e:
        logger.error("Deployment failed: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()

