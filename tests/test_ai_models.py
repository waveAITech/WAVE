import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json
import logging
from wave_ai.lifecycle_manager import AILifecycleManager
from wave_ai.models import CRISPRModel, ProteinFoldingModel, EpigeneticModel
from wave_ai.data_ingestion import DataIngestionLayer
from wave_ai.security import SecurityLayer
from wave_ai.governance import GovernanceFramework

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestAIModels(unittest.TestCase):
    def setUp(self):
        logger.debug("Setting up TestAIModels")
        # Initialize components
        self.data_ingestion = DataIngestionLayer()
        self.security = SecurityLayer()
        self.governance = GovernanceFramework()
        self.lifecycle_manager = AILifecycleManager(
            data_layer=self.data_ingestion,
            security_layer=self.security,
            governance_framework=self.governance
        )
        # Mock models
        self.crispr_model = CRISPRModel()
        self.protein_folding_model = ProteinFoldingModel()
        self.epigenetic_model = EpigeneticModel()
        self.models = [self.crispr_model, self.protein_folding_model, self.epigenetic_model]
        self.lifecycle_manager.register_models(self.models)
        logger.debug("Components initialized and models registered")

    @patch('wave_ai.models.CRISPRModel.evaluate')
    @patch('wave_ai.models.ProteinFoldingModel.evaluate')
    @patch('wave_ai.models.EpigeneticModel.evaluate')
    def test_model_evaluation(self, mock_epi_eval, mock_protein_eval, mock_crispr_eval):
        logger.debug("Testing model evaluation")
        # Setup mock return values
        mock_crispr_eval.return_value = {'precision': 0.95, 'viability': 0.90, 'risk': 0.05}
        mock_protein_eval.return_value = {'precision': 0.92, 'viability': 0.88, 'risk': 0.04}
        mock_epi_eval.return_value = {'precision': 0.93, 'viability': 0.89, 'risk': 0.03}

        # Run evaluation
        evaluation_results = self.lifecycle_manager.evaluate_models()

        # Assertions
        self.assertIn('CRISPRModel', evaluation_results)
        self.assertIn('ProteinFoldingModel', evaluation_results)
        self.assertIn('EpigeneticModel', evaluation_results)

        self.assertEqual(evaluation_results['CRISPRModel']['precision'], 0.95)
        self.assertEqual(evaluation_results['ProteinFoldingModel']['viability'], 0.88)
        self.assertEqual(evaluation_results['EpigeneticModel']['risk'], 0.03)
        logger.debug("Model evaluation test passed")

    @patch('wave_ai.data_ingestion.DataIngestionLayer.load_genomic_data')
    def test_data_ingestion(self, mock_load_genomic):
        logger.debug("Testing data ingestion")
        # Mock data
        mock_data = {
            "gene_sequences": ["ATCG", "GCTA"],
            "clinical_trials": [{"id": 1, "outcome": "success"}],
            "environmental_data": {"soil_quality": 85}
        }
        mock_load_genomic.return_value = mock_data

        # Ingest data
        data = self.data_ingestion.load_genomic_data()

        # Assertions
        self.assertIn("gene_sequences", data)
        self.assertIn("clinical_trials", data)
        self.assertIn("environmental_data", data)
        self.assertEqual(len(data["gene_sequences"]), 2)
        self.assertEqual(data["environmental_data"]["soil_quality"], 85)
        logger.debug("Data ingestion test passed")

    @patch('wave_ai.security.SecurityLayer.verify_compliance')
    def test_security_compliance(self, mock_verify):
        logger.debug("Testing security compliance")
        # Mock compliance
        mock_verify.return_value = True

        # Verify compliance
        is_compliant = self.security.verify_compliance("CRISPR-Cas9")

        # Assertions
        self.assertTrue(is_compliant)
        mock_verify.assert_called_with("CRISPR-Cas9")
        logger.debug("Security compliance test passed")

    @patch('wave_ai.governance.GovernanceFramework.audit_model_history')
    def test_governance_audit(self, mock_audit):
        logger.debug("Testing governance audit")
        # Mock audit data
        mock_audit.return_value = {
            "model_changes": [{"model": "CRISPRModel", "change": "v1.2"}],
            "experiments": [{"id": 101, "outcome": "positive"}]
        }

        # Audit governance
        audit_report = self.governance.audit_model_history()

        # Assertions
        self.assertIn("model_changes", audit_report)
        self.assertIn("experiments", audit_report)
        self.assertEqual(audit_report["model_changes"][0]["model"], "CRISPRModel")
        self.assertEqual(audit_report["experiments"][0]["outcome"], "positive")
        logger.debug("Governance audit test passed")

    def test_lifecycle_manager_integration(self):
        logger.debug("Testing lifecycle manager integration")
        with patch.object(self.lifecycle_manager, 'evaluate_models', return_value={
            'CRISPRModel': {'precision': 0.95, 'viability': 0.90, 'risk': 0.05},
            'ProteinFoldingModel': {'precision': 0.92, 'viability': 0.88, 'risk': 0.04},
            'EpigeneticModel': {'precision': 0.93, 'viability': 0.89, 'risk': 0.03}
        }) as mock_evaluate:
            selected_model = self.lifecycle_manager.select_top_model()
            self.assertEqual(selected_model, self.crispr_model)
            mock_evaluate.assert_called_once()
        logger.debug("Lifecycle manager integration test passed")

    def tearDown(self):
        logger.debug("Tearing down TestAIModels")
        del self.lifecycle_manager
        del self.data_ingestion
        del self.security
        del self.governance
        logger.debug("Teardown complete")

if __name__ == '__main__':
    unittest.main()

