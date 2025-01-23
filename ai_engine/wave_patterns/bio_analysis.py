import os
import json
import logging
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from gseapy import enrichr
from sqlalchemy.exc import SQLAlchemyError

from wave_ai.config import ConfigManager
from wave_ai.utils import DatabaseConnector, APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VariantEffectAnalyzer:
    """
    Analyzes the effects of genetic variants on gene function and phenotype.
    Utilizes external annotation tools and databases to predict variant impacts.
    """

    def __init__(self, config_path: str = 'config/bio_analysis_config.json'):
        logger.info("Initializing VariantEffectAnalyzer with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.annotation_tool = self.config.get('variant_effect_annotation', {}).get('tool', 'snpEff')
        self.reference_genome = self.config.get('variant_effect_annotation', {}).get('reference_genome')
        if not self.reference_genome:
            logger.error("Reference genome must be specified in the configuration.")
            raise ValueError("Reference genome must be specified in the configuration.")
        self.output_dir = self.config.get('variant_effect_annotation', {}).get('output_dir', 'analysis/variant_effect/')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("VariantEffectAnalyzer initialized successfully")

    def fetch_variants(self) -> pd.DataFrame:
        """
        Fetches genetic variant data from the database.
        """
        logger.info("Fetching genetic variants from the database")
        query = self.config.get('variant_effect_annotation', {}).get('query')
        if not query:
            logger.error("No SQL query found for variant effect annotation.")
            raise ValueError("No SQL query found for variant effect annotation.")
        try:
            data = self.db_connector.execute_query(query)
            df_variants = pd.DataFrame(data)
            logger.info("Fetched %d genetic variants for annotation", len(df_variants))
            return df_variants
        except SQLAlchemyError as e:
            logger.error("Database error while fetching variants: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error while fetching variants: %s", e)
            raise e

    def annotate_variants(self, df_variants: pd.DataFrame) -> pd.DataFrame:
        """
        Annotates genetic variants using the specified annotation tool.
        """
        logger.info("Annotating genetic variants using %s", self.annotation_tool)
        try:
            if self.annotation_tool.lower() == 'snpEff':
                annotated_variants = self._annotate_with_snpeff(df_variants)
            elif self.annotation_tool.lower() == 'vep':
                annotated_variants = self._annotate_with_vep(df_variants)
            else:
                logger.error("Unsupported annotation tool: %s", self.annotation_tool)
                raise ValueError(f"Unsupported annotation tool: {self.annotation_tool}")
            logger.info("Variant annotation completed successfully")
            return annotated_variants
        except Exception as e:
            logger.error("Error during variant annotation: %s", e)
            raise e

    def _annotate_with_snpeff(self, df_variants: pd.DataFrame) -> pd.DataFrame:
        """
        Annotates variants using SnpEff.
        Assumes that SnpEff is installed and configured.
        """
        logger.info("Annotating variants with SnpEff")
        try:
            # Prepare VCF file for annotation
            vcf_path = os.path.join(self.output_dir, 'variants.vcf')
            self._create_vcf(df_variants, vcf_path)
            
            # Run SnpEff
            snpeff_cmd = f"snpEff {self.reference_genome} {vcf_path} > {os.path.join(self.output_dir, 'annotated_variants.vcf')}"
            logger.debug("Running SnpEff command: %s", snpeff_cmd)
            os.system(snpeff_cmd)
            
            # Parse annotated VCF
            annotated_df = self._parse_annotated_vcf(os.path.join(self.output_dir, 'annotated_variants.vcf'))
            return annotated_df
        except Exception as e:
            logger.error("Error annotating variants with SnpEff: %s", e)
            raise e

    def _annotate_with_vep(self, df_variants: pd.DataFrame) -> pd.DataFrame:
        """
        Annotates variants using Ensembl VEP.
        Assumes that VEP is installed and configured.
        """
        logger.info("Annotating variants with Ensembl VEP")
        try:
            # Prepare VCF file for annotation
            vcf_path = os.path.join(self.output_dir, 'variants.vcf')
            self._create_vcf(df_variants, vcf_path)
            
            # Run VEP
            vep_cmd = f"vep -i {vcf_path} --cache --dir_cache /path/to/vep/cache --output_file {os.path.join(self.output_dir, 'annotated_variants.vep.vcf')} --assembly GRCh38 --fields Consequence,IMPACT,SYMBOL"
            logger.debug("Running VEP command: %s", vep_cmd)
            os.system(vep_cmd)
            
            # Parse annotated VCF
            annotated_df = self._parse_annotated_vep_vcf(os.path.join(self.output_dir, 'annotated_variants.vep.vcf'))
            return annotated_df
        except Exception as e:
            logger.error("Error annotating variants with VEP: %s", e)
            raise e

    def _create_vcf(self, df_variants: pd.DataFrame, vcf_path: str):
        """
        Creates a VCF file from the variants DataFrame.
        """
        logger.info("Creating VCF file at %s", vcf_path)
        try:
            with open(vcf_path, 'w') as vcf_file:
                # Write VCF headers
                vcf_file.write("##fileformat=VCFv4.2\n")
                vcf_file.write("##INFO=<ID=ANN,Number=.,Type=String,Description=\"Annotation from SnpEff\">\n")
                vcf_file.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
                # Write variant entries
                for _, row in df_variants.iterrows():
                    chrom = row['chromosome']
                    pos = row['position']
                    var_id = row.get('variant_id', '.')
                    ref = row['reference']
                    alt = row['alternate']
                    qual = row.get('quality', '.')
                    filt = row.get('filter', 'PASS')
                    info = f"ANN={row.get('annotation', '.')}"
                    vcf_file.write(f"{chrom}\t{pos}\t{var_id}\t{ref}\t{alt}\t{qual}\t{filt}\t{info}\n")
            logger.info("VCF file created successfully at %s", vcf_path)
        except Exception as e:
            logger.error("Error creating VCF file: %s", e)
            raise e

    def _parse_annotated_vcf(self, annotated_vcf_path: str) -> pd.DataFrame:
        """
        Parses the annotated VCF file from SnpEff.
        """
        logger.info("Parsing annotated VCF from SnpEff at %s", annotated_vcf_path)
        try:
            annotated_variants = []
            with open(annotated_vcf_path, 'r') as vcf_file:
                for line in vcf_file:
                    if line.startswith('#'):
                        continue
                    fields = line.strip().split('\t')
                    chrom, pos, var_id, ref, alt, qual, filt, info = fields[:8]
                    annotations = info.split('ANN=')[-1].split(';')[0]
                    annotated_variants.append({
                        'chromosome': chrom,
                        'position': pos,
                        'variant_id': var_id,
                        'reference': ref,
                        'alternate': alt,
                        'quality': qual,
                        'filter': filt,
                        'annotation': annotations
                    })
            df_annotated = pd.DataFrame(annotated_variants)
            logger.info("Parsed %d annotated variants from SnpEff VCF", len(df_annotated))
            return df_annotated
        except Exception as e:
            logger.error("Error parsing annotated VCF: %s", e)
            raise e

    def _parse_annotated_vep_vcf(self, annotated_vep_vcf_path: str) -> pd.DataFrame:
        """
        Parses the annotated VCF file from VEP.
        """
        logger.info("Parsing annotated VCF from VEP at %s", annotated_vep_vcf_path)
        try:
            annotated_variants = []
            with open(annotated_vep_vcf_path, 'r') as vcf_file:
                for line in vcf_file:
                    if line.startswith('#'):
                        continue
                    fields = line.strip().split('\t')
                    chrom, pos, var_id, ref, alt, qual, filt, info = fields[:8]
                    annotations = {}
                    info_fields = info.split(';')
                    for info_field in info_fields:
                        if info_field.startswith('Consequence='):
                            annotations['Consequence'] = info_field.split('=')[1]
                        elif info_field.startswith('IMPACT='):
                            annotations['Impact'] = info_field.split('=')[1]
                        elif info_field.startswith('SYMBOL='):
                            annotations['Gene'] = info_field.split('=')[1]
                    annotated_variants.append({
                        'chromosome': chrom,
                        'position': pos,
                        'variant_id': var_id,
                        'reference': ref,
                        'alternate': alt,
                        'quality': qual,
                        'filter': filt,
                        'consequence': annotations.get('Consequence', '.'),
                        'impact': annotations.get('Impact', '.'),
                        'gene': annotations.get('Gene', '.')
                    })
            df_annotated = pd.DataFrame(annotated_variants)
            logger.info("Parsed %d annotated variants from VEP VCF", len(df_annotated))
            return df_annotated
        except Exception as e:
            logger.error("Error parsing annotated VEP VCF: %s", e)
            raise e

    def save_annotations(self, df_annotated: pd.DataFrame):
        """
        Saves the annotated variants back to the database.
        """
        logger.info("Saving annotated variants to the database")
        table_name = self.config.get('variant_effect_annotation', {}).get('output_table', 'annotated_variants')
        try:
            self.db_connector.insert_dataframe(table_name, df_annotated)
            logger.info("Annotated variants saved successfully to table: %s", table_name)
        except SQLAlchemyError as e:
            logger.error("Database error while saving annotated variants: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error while saving annotated variants: %s", e)
            raise e

    def run_analysis(self):
        """
        Executes the full variant effect analysis pipeline.
        """
        logger.info("Starting variant effect analysis pipeline")
        try:
            df_variants = self.fetch_variants()
            df_annotated = self.annotate_variants(df_variants)
            self.save_annotations(df_annotated)
            logger.info("Variant effect analysis pipeline completed successfully")
        except Exception as e:
            logger.error("Variant effect analysis pipeline failed: %s", e)
            raise e


class PathwayEnrichmentAnalyzer:
    """
    Performs pathway enrichment analysis on gene lists to identify biological pathways
    that are significantly overrepresented.
    """

    def __init__(self, config_path: str = 'config/bio_analysis_config.json'):
        logger.info("Initializing PathwayEnrichmentAnalyzer with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.enrichment_tool = self.config.get('pathway_enrichment', {}).get('tool', 'enrichr')
        self.reference_geneset_library = self.config.get('pathway_enrichment', {}).get('geneset_library', 'KEGG_2021_Human')
        self.output_dir = self.config.get('pathway_enrichment', {}).get('output_dir', 'analysis/pathway_enrichment/')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("PathwayEnrichmentAnalyzer initialized successfully")

    def fetch_gene_list(self) -> List[str]:
        """
        Fetches the list of genes for enrichment analysis from the database.
        """
        logger.info("Fetching gene list for pathway enrichment analysis")
        query = self.config.get('pathway_enrichment', {}).get('query')
        if not query:
            logger.error("No SQL query found for pathway enrichment analysis.")
            raise ValueError("No SQL query found for pathway enrichment analysis.")
        try:
            data = self.db_connector.execute_query(query)
            df_genes = pd.DataFrame(data)
            gene_list = df_genes['gene_symbol'].dropna().unique().tolist()
            logger.info("Fetched %d unique genes for enrichment analysis", len(gene_list))
            return gene_list
        except SQLAlchemyError as e:
            logger.error("Database error while fetching gene list: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error while fetching gene list: %s", e)
            raise e

    def perform_enrichment(self, gene_list: List[str]) -> pd.DataFrame:
        """
        Performs pathway enrichment analysis using the specified tool.
        """
        logger.info("Performing pathway enrichment analysis using %s", self.enrichment_tool)
        try:
            if self.enrichment_tool.lower() == 'enrichr':
                enrichment_results = enrichr(gene_list=gene_list,
                                             gene_sets=[self.reference_geneset_library],
                                             organism='Human',
                                             outdir=self.output_dir,
                                             cutoff=0.05)
                df_enrichr = enrichment_results.results[self.reference_geneset_library]
                logger.info("Pathway enrichment analysis with Enrichr completed successfully")
                return df_enrichr
            else:
                logger.error("Unsupported enrichment tool: %s", self.enrichment_tool)
                raise ValueError(f"Unsupported enrichment tool: {self.enrichment_tool}")
        except Exception as e:
            logger.error("Error during pathway enrichment analysis: %s", e)
            raise e

    def save_enrichment_results(self, df_enrichment: pd.DataFrame):
        """
        Saves the enrichment analysis results to the database.
        """
        logger.info("Saving pathway enrichment results to the database")
        table_name = self.config.get('pathway_enrichment', {}).get('output_table', 'pathway_enrichment_results')
        try:
            self.db_connector.insert_dataframe(table_name, df_enrichment)
            logger.info("Pathway enrichment results saved successfully to table: %s", table_name)
        except SQLAlchemyError as e:
            logger.error("Database error while saving enrichment results: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error while saving enrichment results: %s", e)
            raise e

    def run_analysis(self):
        """
        Executes the full pathway enrichment analysis pipeline.
        """
        logger.info("Starting pathway enrichment analysis pipeline")
        try:
            gene_list = self.fetch_gene_list()
            if not gene_list:
                logger.warning("No genes available for enrichment analysis.")
                return
            df_enrichment = self.perform_enrichment(gene_list)
            self.save_enrichment_results(df_enrichment)
            logger.info("Pathway enrichment analysis pipeline completed successfully")
        except Exception as e:
            logger.error("Pathway enrichment analysis pipeline failed: %s", e)
            raise e


class GeneExpressionAnalyzer:
    """
    Analyzes gene expression data to identify differentially expressed genes,
    expression patterns, and correlations with phenotypic traits.
    """

    def __init__(self, config_path: str = 'config/bio_analysis_config.json'):
        logger.info("Initializing GeneExpressionAnalyzer with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.db_connector = DatabaseConnector(self.config.get('database'))
        self.api_client = APIClient(self.config.get('api'))
        self.output_dir = self.config.get('gene_expression_analysis', {}).get('output_dir', 'analysis/gene_expression/')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("GeneExpressionAnalyzer initialized successfully")

    def fetch_expression_data(self) -> pd.DataFrame:
        """
        Fetches gene expression data from the database.
        """
        logger.info("Fetching gene expression data from the database")
        query = self.config.get('gene_expression_analysis', {}).get('query')
        if not query:
            logger.error("No SQL query found for gene expression analysis.")
            raise ValueError("No SQL query found for gene expression analysis.")
        try:
            data = self.db_connector.execute_query(query)
            df_expression = pd.DataFrame(data)
            logger.info("Fetched gene expression data with %d records", len(df_expression))
            return df_expression
        except SQLAlchemyError as e:
            logger.error("Database error while fetching gene expression data: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error while fetching gene expression data: %s", e)
            raise e

    def identify_differentially_expressed_genes(self, df_expression: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies differentially expressed genes between conditions using statistical tests.
        """
        logger.info("Identifying differentially expressed genes")
        try:
            condition_col = self.config.get('gene_expression_analysis', {}).get('condition_column')
            gene_col = self.config.get('gene_expression_analysis', {}).get('gene_column', 'gene_symbol')
            expression_col = self.config.get('gene_expression_analysis', {}).get('expression_column', 'expression_level')

            if not condition_col or condition_col not in df_expression.columns:
                logger.error("Condition column '%s' not found in expression data", condition_col)
                raise ValueError(f"Condition column '{condition_col}' not found in expression data")

            conditions = df_expression[condition_col].unique()
            if len(conditions) != 2:
                logger.error("Differential expression analysis requires exactly two conditions.")
                raise ValueError("Differential expression analysis requires exactly two conditions.")

            condition_a, condition_b = conditions
            logger.debug("Conditions identified: %s vs %s", condition_a, condition_b)

            genes = df_expression[gene_col].unique()
            diff_expr_results = []

            for gene in genes:
                expr_a = df_expression[df_expression[gene_col] == gene][expression_col][df_expression[condition_col] == condition_a]
                expr_b = df_expression[df_expression[gene_col] == gene][expression_col][df_expression[condition_col] == condition_b]

                if len(expr_a) < 3 or len(expr_b) < 3:
                    logger.debug("Not enough samples for gene: %s. Skipping.", gene)
                    continue

                # Perform t-test
                from scipy.stats import ttest_ind
                t_stat, p_val = ttest_ind(expr_a, expr_b, equal_var=False)
                fold_change = expr_b.mean() / expr_a.mean() if expr_a.mean() != 0 else np.nan

                diff_expr_results.append({
                    'gene': gene,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'fold_change': fold_change
                })

            df_diff_expr = pd.DataFrame(diff_expr_results)
            # Adjust p-values for multiple testing
            from statsmodels.stats.multitest import multipletests
            df_diff_expr['adj_p_value'] = multipletests(df_diff_expr['p_value'], method='fdr_bh')[1]

            # Filter significant genes
            significance_threshold = self.config.get('gene_expression_analysis', {}).get('significance_threshold', 0.05)
            df_significant = df_diff_expr[df_diff_expr['adj_p_value'] < significance_threshold]
            logger.info("Identified %d differentially expressed genes", len(df_significant))
            return df_significant

    def correlate_expression_with_phenotype(self, df_expression: pd.DataFrame) -> pd.DataFrame:
        """
        Correlates gene expression levels with phenotypic traits.
        """
        logger.info("Correlating gene expression with phenotypic traits")
        try:
            phenotype_col = self.config.get('gene_expression_analysis', {}).get('phenotype_column')
            gene_col = self.config.get('gene_expression_analysis', {}).get('gene_column', 'gene_symbol')
            expression_col = self.config.get('gene_expression_analysis', {}).get('expression_column', 'expression_level')

            if not phenotype_col or phenotype_col not in df_expression.columns:
                logger.error("Phenotype column '%s' not found in expression data", phenotype_col)
                raise ValueError(f"Phenotype column '{phenotype_col}' not found in expression data")

            genes = df_expression[gene_col].unique()
            correlations = []

            for gene in genes:
                expr = df_expression[df_expression[gene_col] == gene][expression_col]
                phenotype = df_expression[df_expression[gene_col] == gene][phenotype_col]

                if len(expr) < 2 or len(phenotype) < 2:
                    logger.debug("Not enough data points for gene: %s. Skipping.", gene)
                    continue

                correlation = expr.corr(phenotype)
                correlations.append({
                    'gene': gene,
                    'correlation_coefficient': correlation
                })

            df_correlations = pd.DataFrame(correlations)
            logger.info("Correlated gene expression with phenotypic traits successfully")
            return df_correlations
        except Exception as e:
            logger.error("Error correlating gene expression with phenotype: %s", e)
            raise e

    def visualize_differential_expression(self, df_diff_expr: pd.DataFrame):
        """
        Generates a volcano plot for differentially expressed genes.
        """
        logger.info("Generating volcano plot for differentially expressed genes")
        try:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df_diff_expr, x='fold_change', y=-np.log10(df_diff_expr['adj_p_value']), hue=(df_diff_expr['fold_change'] > 1.5) & (df_diff_expr['adj_p_value'] < 0.05), palette={True: 'red', False: 'grey'}, edgecolor=None, alpha=0.7)
            plt.title('Volcano Plot of Differentially Expressed Genes')
            plt.xlabel('Fold Change')
            plt.ylabel('-Log10 Adjusted P-Value')
            plt.axvline(x=1.5, color='blue', linestyle='--')
            plt.axhline(y=-np.log10(0.05), color='blue', linestyle='--')
            plt.legend(title='Significant', loc='upper right')
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'volcano_plot.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info("Volcano plot saved at %s", plot_path)
        except Exception as e:
            logger.error("Error generating volcano plot: %s", e)
            raise e

    def save_results(self, df_diff_expr: pd.DataFrame, df_correlations: pd.DataFrame):
        """
        Saves the analysis results to the database.
        """
        logger.info("Saving analysis results to the database")
        try:
            self.db_connector.insert_dataframe('differentially_expressed_genes', df_diff_expr)
            self.db_connector.insert_dataframe('gene_phenotype_correlations', df_correlations)
            logger.info("Analysis results saved successfully to the database")
        except SQLAlchemyError as e:
            logger.error("Database error while saving analysis results: %s", e)
            raise e
        except Exception as e:
            logger.error("Unexpected error while saving analysis results: %s", e)
            raise e

    def run_analysis(self):
        """
        Executes the full gene expression analysis pipeline.
        """
        logger.info("Starting gene expression analysis pipeline")
        try:
            df_expression = self.fetch_expression_data()
            df_diff_expr = self.identify_differentially_expressed_genes(df_expression)
            df_correlations = self.correlate_expression_with_phenotype(df_expression)
            self.visualize_differential_expression(df_diff_expr)
            self.save_results(df_diff_expr, df_correlations)
            logger.info("Gene expression analysis pipeline completed successfully")
        except Exception as e:
            logger.error("Gene expression analysis pipeline failed: %s", e)
            raise e


class BioAnalysisManager:
    """
    Manages the overall biological analysis within the WAVE ecosystem,
    coordinating variant effect analysis, pathway enrichment, and gene expression analysis.
    """

    def __init__(self, config_path: str = 'config/bio_analysis_config.json'):
        logger.info("Initializing BioAnalysisManager with config path: %s", config_path)
        self.config = ConfigManager(config_path)
        self.variant_effect_analyzer = VariantEffectAnalyzer(config_path=config_path)
        self.pathway_enrichment_analyzer = PathwayEnrichmentAnalyzer(config_path=config_path)
        self.gene_expression_analyzer = GeneExpressionAnalyzer(config_path=config_path)
        logger.info("BioAnalysisManager initialized successfully")

    def run_full_bio_analysis(self):
        """
        Executes the full biological analysis pipeline.
        """
        logger.info("Running full biological analysis pipeline")
        try:
            # Variant Effect Analysis
            self.variant_effect_analyzer.run_analysis()

            # Pathway Enrichment Analysis
            self.pathway_enrichment_analyzer.run_analysis()

            # Gene Expression Analysis
            self.gene_expression_analyzer.run_analysis()

            logger.info("Full biological analysis pipeline completed successfully")
        except Exception as e:
            logger.error("Full biological analysis pipeline failed: %s", e)
            raise e


def main():
    import argparse

    parser = argparse.ArgumentParser(description='WAVE Biological Analysis Module')
    parser.add_argument('--config', type=str, default='config/bio_analysis_config.json', help='Path to biological analysis configuration file')
    parser.add_argument('--run_full', action='store_true', help='Run the full biological analysis pipeline')
    args = parser.parse_args()

    try:
        manager = BioAnalysisManager(config_path=args.config)
        if args.run_full:
            manager.run_full_bio_analysis()
        else:
            logger.error("No action specified. Use --run_full to execute the full analysis pipeline.")
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error("Biological analysis failed: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()

