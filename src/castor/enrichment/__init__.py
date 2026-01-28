"""Downstream enrichment analysis: gene contributions and pathway enrichment."""

from castor.enrichment.gene_analysis import identify_contributing_genes
from castor.enrichment.pathway import fetch_orthologs, run_pathway_enrichment

__all__ = [
    "identify_contributing_genes",
    "fetch_orthologs",
    "run_pathway_enrichment",
]
