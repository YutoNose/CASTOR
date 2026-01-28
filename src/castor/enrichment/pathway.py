"""GO/KEGG pathway enrichment analysis via GSEA."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


_CACHE_DIR = Path.home() / ".cache" / "castor"
_ORTHO_CACHE = _CACHE_DIR / "orthologs_mouse_human.csv"


def fetch_orthologs(
    output_path: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch mouse-to-human ortholog mapping from Ensembl BioMart.

    Results are cached locally at ``~/.cache/castor/orthologs_mouse_human.csv``
    so subsequent calls avoid network requests.

    Parameters
    ----------
    output_path : str | None
        Save mapping CSV to this path (in addition to cache).
    use_cache : bool
        If *True* (default), reuse a previously downloaded mapping.

    Returns
    -------
    pd.DataFrame
        Columns: ``mouse_gene``, ``human_gene``.
    """
    # Try local cache first
    if use_cache and _ORTHO_CACHE.exists():
        logger.info("Loading cached orthologs from %s", _ORTHO_CACHE)
        return pd.read_csv(_ORTHO_CACHE)

    xml_query = """
    <!DOCTYPE Query>
    <Query virtualSchemaName="default" formatter="CSV" header="0"
           uniqueRows="1" count="" datasetConfigVersion="0.6">
        <Dataset name="mmusculus_gene_ensembl" interface="default">
            <Attribute name="external_gene_name" />
            <Attribute name="hsapiens_homolog_associated_gene_name" />
        </Dataset>
    </Query>
    """

    url = "http://www.ensembl.org/biomart/martservice"
    r = requests.get(url, params={"query": xml_query}, timeout=60)
    r.raise_for_status()

    if "Query ERROR" in r.text:
        raise RuntimeError("BioMart Query Error: " + r.text)

    lines = r.text.strip().split("\n")
    data = [line.split(",") for line in lines if line.strip()]

    df = pd.DataFrame(data, columns=["mouse_gene", "human_gene"])
    df = df.dropna()
    df = df[df["human_gene"] != ""]

    # Save to cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_ORTHO_CACHE, index=False)
    logger.info("Orthologs cached to %s (%d mappings)", _ORTHO_CACHE, len(df))

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def run_pathway_enrichment(
    gene_ranking: pd.DataFrame,
    species: str = "human",
    database: str = "GO_Biological_Process_2023",
    orthologs: pd.DataFrame | None = None,
    min_size: int = 3,
    max_size: int = 2000,
    permutation_num: int = 1000,
    rank_col: str = "Fold_Change",
) -> pd.DataFrame:
    """Run GSEA-based pathway enrichment.

    Parameters
    ----------
    gene_ranking : pd.DataFrame
        Must have ``Gene`` and *rank_col* columns.
    species : str
        ``"human"`` or ``"mouse"`` (requires *orthologs* for mouse).
    database : str
        Enrichr library name.
    orthologs : pd.DataFrame | None
        Ortholog mapping (required when *species* is ``"mouse"``).
    min_size, max_size : int
        Gene-set size filter.
    permutation_num : int
        Number of GSEA permutations.
    rank_col : str
        Column used for ranking.

    Returns
    -------
    pd.DataFrame
        Columns: ``Pathway``, ``Enrichment_Score``, ``P_value``,
        ``Q_value``, ``Genes``, ``Gene_Count``.
    """
    import gseapy as gp

    # Suppress noisy gseapy ERROR logs (they duplicate our own warnings)
    logging.getLogger("gseapy").setLevel(logging.CRITICAL)

    empty = pd.DataFrame(
        columns=["Pathway", "Enrichment_Score", "P_value", "Q_value", "Genes", "Gene_Count"]
    )

    df = gene_ranking.copy()

    if rank_col not in df.columns and "Score" in df.columns:
        rank_col = "Score"

    if "Gene" not in df.columns or rank_col not in df.columns:
        raise ValueError(f"gene_ranking must have 'Gene' and '{rank_col}' columns")

    # Mouse → human ortholog mapping (if available), then uppercase for Enrichr
    if species.lower() == "mouse" and orthologs is not None:
        mapping = dict(
            zip(orthologs["mouse_gene"].str.upper(), orthologs["human_gene"].str.upper())
        )
        df["Gene"] = df["Gene"].str.upper().map(mapping).fillna(df["Gene"].str.upper())
    else:
        # Enrichr gene sets use uppercase symbols; uppercase handles both
        # human genes and mouse genes whose symbols match human orthologs
        df["Gene"] = df["Gene"].str.upper()

    df = df.groupby("Gene")[rank_col].max().reset_index()

    rng = np.random.default_rng(42)
    df[rank_col] = df[rank_col] + rng.normal(0, 1e-6, size=len(df))

    rnk = df.sort_values(rank_col, ascending=False)[["Gene", rank_col]]

    if len(rnk) < 3:
        return empty

    try:
        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets=database,
            threads=4,
            min_size=min_size,
            max_size=max_size,
            permutation_num=permutation_num,
            outdir=None,
            seed=42,
            verbose=False,
        )

        res = pre_res.res2d.copy()
        if res.empty:
            return empty

        res = res.rename(
            columns={
                "Term": "Pathway",
                "NES": "Enrichment_Score",
                "NOM p-val": "P_value",
                "FDR q-val": "Q_value",
                "Lead_genes": "Genes",
            }
        )

        for col in ("P_value", "Q_value", "Enrichment_Score"):
            if col in res.columns:
                res[col] = pd.to_numeric(res[col], errors="coerce")

        res["Genes"] = res["Genes"].astype(str).str.replace(";", ",")
        res["Gene_Count"] = res["Genes"].apply(lambda x: len(x.split(",")) if x else 0)
        res["Pathway"] = res["Pathway"].apply(
            lambda t: re.sub(r"\s*\(GO:\d+\)|R-HSA-\d+", "", str(t)).strip()
        )

        return res[
            ["Pathway", "Enrichment_Score", "P_value", "Q_value", "Genes", "Gene_Count"]
        ].sort_values("P_value")

    except Exception as exc:
        logger.warning("Enrichment failed: %s", exc)
        return empty
