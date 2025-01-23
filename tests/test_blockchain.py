import unittest
from unittest.mock import patch, MagicMock
import logging
from wave_ai.blockchain import Blockchain, Block
from wave_ai.governance import GovernanceFramework

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestBlockchain(unittest.TestCase):
    def setUp(self):
        logger.debug("Setting up TestBlockchain")
        # Initialize the blockchain and governance framework
        self.blockchain = Blockchain()
        self.governance = GovernanceFramework()
        logger.debug("Blockchain and GovernanceFramework initialized")

    def test_initial_block(self):
        logger.debug("Testing initial block creation")
        initial_block = self.blockchain.chain[0]
        self.assertEqual(initial_block.index, 0, "Initial block index should be 0")
        self.assertEqual(initial_block.previous_hash, "0", "Initial block previous_hash should be '0'")
        self.assertIsNotNone(initial_block.hash, "Initial block should have a valid hash")
        logger.debug("Initial block test passed")

    def test_add_block(self):
        logger.debug("Testing adding a new block to the blockchain")
        previous_block = self.blockchain.get_last_block()
        new_block = self.blockchain.create_block(data={"action": "add_model", "model_id": "CRISPRModel_v1.2"})
        self.blockchain.add_block(new_block)
        self.assertEqual(new_block.index, previous_block.index + 1, "New block index should be previous index + 1")
        self.assertEqual(new_block.previous_hash, previous_block.hash, "New block's previous_hash should match last block's hash")
        self.assertTrue(self.blockchain.is_chain_valid(), "Blockchain should be valid after adding a new block")
        logger.debug("Add block test passed")

    def test_chain_integrity(self):
        logger.debug("Testing blockchain integrity")
        # Add a legitimate block
        new_block = self.blockchain.create_block(data={"action": "update_model", "model_id": "ProteinFoldingModel_v1.3"})
        self.blockchain.add_block(new_block)
        self.assertTrue(self.blockchain.is_chain_valid(), "Blockchain should be valid after adding a legitimate block")

        # Tamper with the blockchain
        logger.debug("Tampering with the blockchain to test integrity")
        self.blockchain.chain[1].data = {"action": "tampered_action"}
        self.assertFalse(self.blockchain.is_chain_valid(), "Blockchain should be invalid after tampering")
        logger.debug("Chain integrity test passed")

    @patch('wave_ai.blockchain.Block.compute_hash')
    def test_block_hashing(self, mock_compute_hash):
        logger.debug("Testing block hashing mechanism")
        mock_compute_hash.return_value = "fake_hash"
        block = self.blockchain.create_block(data={"action": "test_hashing"})
        self.assertEqual(block.hash, "fake_hash", "Block hash should match the mocked hash")
        mock_compute_hash.assert_called_once_with()
        logger.debug("Block hashing test passed")

    def test_governance_integration(self):
        logger.debug("Testing integration between governance framework and blockchain")
        with patch.object(self.governance, 'audit_model_history') as mock_audit:
            mock_audit.return_value = {
                "model_changes": [{"model": "CRISPRModel", "change": "v1.2"}],
                "experiments": [{"id": 101, "outcome": "positive"}]
            }
            audit_data = self.governance.audit_model_history()
            self.assertIn("model_changes", audit_data, "Audit data should contain 'model_changes'")
            self.assertIn("experiments", audit_data, "Audit data should contain 'experiments'")
            new_block = self.blockchain.create_block(data=audit_data)
            self.blockchain.add_block(new_block)
            self.assertTrue(self.blockchain.is_chain_valid(), "Blockchain should be valid after adding governance audit block")
            logger.debug("Governance integration test passed")

    def test_multiple_block_additions(self):
        logger.debug("Testing multiple block additions to the blockchain")
        actions = [
            {"action": "add_model", "model_id": "EpigeneticModel_v2.0"},
            {"action": "update_model", "model_id": "CRISPRModel_v1.3"},
            {"action": "remove_model", "model_id": "ProteinFoldingModel_v1.1"}
        ]
        for action in actions:
            block = self.blockchain.create_block(data=action)
            self.blockchain.add_block(block)
        
        self.assertEqual(len(self.blockchain.chain), 4, "Blockchain should contain 4 blocks (1 genesis + 3 added)")
        self.assertTrue(self.blockchain.is_chain_valid(), "Blockchain should be valid after multiple additions")
        logger.debug("Multiple block additions test passed")

    def test_block_retrieval(self):
        logger.debug("Testing retrieval of blocks from the blockchain")
        block = self.blockchain.create_block(data={"action": "test_retrieval"})
        self.blockchain.add_block(block)
        retrieved_block = self.blockchain.get_block_by_index(1)
        self.assertEqual(retrieved_block.data, {"action": "test_retrieval"}, "Retrieved block data should match the added block")
        logger.debug("Block retrieval test passed")

    def tearDown(self):
        logger.debug("Tearing down TestBlockchain")
        del self.blockchain
        del self.governance
        logger.debug("Teardown complete")

if __name__ == '__main__':
    unittest.main()

