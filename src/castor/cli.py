"""CASTOR command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(
    name="castor",
    help="CASTOR: Dual-Axis Anomaly Detection for Spatial Transcriptomics",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()

BANNER = """[bold cyan]
  ██████╗ █████╗ ███████╗████████╗ ██████╗ ██████╗
 ██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
 ██║     ███████║███████╗   ██║   ██║   ██║██████╔╝
 ██║     ██╔══██║╚════██║   ██║   ██║   ██║██╔══██╗
 ╚██████╗██║  ██║███████║   ██║   ╚██████╔╝██║  ██║
  ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
[/bold cyan]
[dim]  ⬡ Dual-Axis Anomaly Detection for Spatial Transcriptomics ⬡[/dim]
"""


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


# -----------------------------------------------------------------------
# detect
# -----------------------------------------------------------------------
@app.command()
def detect(
    input_path: Annotated[str, typer.Argument(help="Path to .h5ad or expression CSV")],
    coords: Annotated[
        str | None,
        typer.Option("--coords", "-c", help="Path to coordinates CSV (required for CSV input)"),
    ] = None,
    output_dir: Annotated[
        str, typer.Option("--output-dir", "-o", help="Output directory")
    ] = "castor_output",
    intrinsic: Annotated[
        str,
        typer.Option(
            "--intrinsic",
            "-i",
            help="Intrinsic detection method (pca_error, lof, isolation_forest, mahalanobis, ocsvm)",
        ),
    ] = "pca_error",
    threshold_local: Annotated[
        float, typer.Option("--threshold-local", "-tl", help="Contextual anomaly Z threshold")
    ] = 2.0,
    threshold_global: Annotated[
        float, typer.Option("--threshold-global", "-tg", help="Intrinsic anomaly Z threshold")
    ] = 2.0,
    hidden_dim: Annotated[int, typer.Option("--hidden-dim", help="Model hidden dimension")] = 64,
    n_epochs: Annotated[int, typer.Option("--n-epochs", "-e", help="Training epochs")] = 100,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate")] = 1e-3,
    lambda_pos: Annotated[
        float, typer.Option("--lambda-pos", help="Position prediction loss weight")
    ] = 0.5,
    k_neighbors: Annotated[
        int, typer.Option("--k-neighbors", "-k", help="Spatial graph neighbors")
    ] = 15,
    device: Annotated[str, typer.Option("--device", "-d", help="Device: auto, cpu, cuda")] = "auto",
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    no_plot: Annotated[
        bool, typer.Option("--no-plot", help="Skip generating visualisation")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Minimal output")] = False,
) -> None:
    """[bold cyan]Detect[/bold cyan] anomalies in spatial transcriptomics data.

    Runs the full dual-axis detection pipeline: train inverse prediction
    model, compute contextual and intrinsic scores, and output results.

    [bold]Examples:[/bold]
        castor detect data.h5ad
        castor detect expression.csv --coords coordinates.csv
        castor detect data.h5ad --intrinsic lof --n-epochs 200
    """
    from castor.api import CASTOR
    from castor.config import CASTORConfig
    from castor.io.exporters import save_parameters, save_results

    if not quiet:
        console.print(BANNER)

    logging.basicConfig(
        level=logging.INFO if not quiet else logging.WARNING,
        format="%(message)s",
    )

    cfg = CASTORConfig(
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        learning_rate=lr,
        lambda_pos=lambda_pos,
        k_neighbors=k_neighbors,
        intrinsic_method=intrinsic,
        threshold_local=threshold_local,
        threshold_global=threshold_global,
        device=device,
        random_state=seed,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load + run
    with _progress() as progress:
        task = progress.add_task("Running CASTOR pipeline...", total=None)
        castor = CASTOR(config=cfg)

        inp = Path(input_path)
        if inp.suffix == ".csv" and coords is None:
            console.print(
                Panel("[red]--coords is required for CSV input[/red]", border_style="red")
            )
            raise typer.Exit(1)

        results = castor.fit_predict(
            str(inp),
            coords=coords,
            verbose=not quiet,
        )
        progress.update(task, completed=1, total=1)

    # Save
    save_results(results, out / "results.csv")
    save_parameters(castor.params_, out / "parameters.json")

    # Plot
    if not no_plot:
        try:
            with _progress() as progress:
                task = progress.add_task("Generating plots...", total=None)
                castor.plot_results(save_path=str(out / "castor_6panel.png"))
                progress.update(task, completed=1, total=1)
        except Exception as exc:
            console.print(f"[yellow]Warning: could not generate plots: {exc}[/yellow]")

    # Summary table
    summary = Table(
        title="Detection Results",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold cyan",
    )
    summary.add_column("Diagnosis", style="bold")
    summary.add_column("Count", justify="right", style="green")
    summary.add_column("Percentage", justify="right", style="yellow")

    total = len(results)
    styles = {
        "Normal": "dim",
        "Contextual Anomaly": "blue",
        "Intrinsic Anomaly": "magenta",
        "Confirmed Anomaly": "red bold",
    }

    for diag in ("Normal", "Contextual Anomaly", "Intrinsic Anomaly", "Confirmed Anomaly"):
        count = (results["Diagnosis"] == diag).sum()
        pct = count / total * 100
        s = styles.get(diag, "")
        summary.add_row(f"[{s}]{diag}[/{s}]", f"{count:,}", f"{pct:.1f}%")

    console.print()
    console.print(summary)

    # Output tree
    tree = Tree(f"[bold green]{out}/[/bold green]", guide_style="dim")
    tree.add("[cyan]results.csv[/cyan] -- detection results")
    tree.add("[cyan]parameters.json[/cyan] -- parameters used")
    if not no_plot:
        tree.add("[cyan]castor_6panel.png[/cyan] -- visualisation")

    console.print()
    console.print(Panel(tree, title="Output Files", border_style="green"))


# -----------------------------------------------------------------------
# visualize
# -----------------------------------------------------------------------
@app.command()
def visualize(
    results_csv: Annotated[str, typer.Argument(help="Path to CASTOR results CSV")],
    input_path: Annotated[
        str | None,
        typer.Option("--input", help="Original .h5ad for coordinates"),
    ] = None,
    coords_csv: Annotated[
        str | None,
        typer.Option("--coords", "-c", help="Path to coordinates CSV"),
    ] = None,
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output figure path")
    ] = "castor_6panel.png",
    threshold_local: Annotated[float, typer.Option("--threshold-local")] = 2.0,
    threshold_global: Annotated[float, typer.Option("--threshold-global")] = 2.0,
    dpi: Annotated[int, typer.Option("--dpi")] = 300,
) -> None:
    """[bold cyan]Visualize[/bold cyan] CASTOR results.

    Generates the six-panel figure from a previously saved results CSV.

    [bold]Examples:[/bold]
        castor visualize results.csv --input data.h5ad
        castor visualize results.csv --coords coordinates.csv -o fig.pdf
    """
    import pandas as pd

    from castor.visualization.plots import plot_castor_results

    console.print(BANNER)

    results = pd.read_csv(results_csv, index_col=0)

    # Get coordinates
    if input_path and input_path.endswith(".h5ad"):
        from castor.io.loaders import load_anndata

        _, coords, _ = load_anndata(input_path)
    elif coords_csv:
        coords_df = pd.read_csv(coords_csv, index_col=0)
        coords = coords_df.values
    else:
        console.print(
            Panel(
                "[red]Provide --input (.h5ad) or --coords (.csv) for spatial coordinates[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    with _progress() as progress:
        task = progress.add_task("Generating figure...", total=None)
        plot_castor_results(
            coords,
            results,
            save_path=output,
            dpi=dpi,
            threshold_local=threshold_local,
            threshold_global=threshold_global,
        )
        progress.update(task, completed=1, total=1)

    console.print(Panel(f"[green]Figure saved to: {output}[/green]", border_style="green"))


# -----------------------------------------------------------------------
# enrich
# -----------------------------------------------------------------------
@app.command()
def enrich(
    results_csv: Annotated[str, typer.Argument(help="Path to CASTOR results CSV")],
    expression: Annotated[
        str, typer.Option("--expression", "-e", help="Path to expression data (.h5ad or .csv)")
    ],
    diagnosis_type: Annotated[
        str,
        typer.Option("--diagnosis", "-t", help="Anomaly type to analyse"),
    ] = "Confirmed Anomaly",
    species: Annotated[
        str,
        typer.Option("--species", help="Species: human or mouse"),
    ] = "human",
    database: Annotated[
        str,
        typer.Option("--database", help="Enrichr library name"),
    ] = "GO_Biological_Process_2023",
    top_n: Annotated[
        int,
        typer.Option("--top-n", "-n", help="Number of top genes"),
    ] = 20,
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Output directory"),
    ] = "castor_enrichment",
    no_plot: Annotated[
        bool, typer.Option("--no-plot", help="Skip generating figures")
    ] = False,
    per_spot: Annotated[
        bool, typer.Option("--per-spot", help="Run enrichment per high-scoring spot")
    ] = False,
    top_spots: Annotated[
        int, typer.Option("--top-spots", help="Number of top spots for per-spot enrichment")
    ] = 10,
) -> None:
    """[bold cyan]Enrich[/bold cyan] -- gene contribution and pathway enrichment.

    Identifies genes driving detected anomalies and runs GSEA-based
    pathway enrichment analysis.

    [bold]Examples:[/bold]
        castor enrich results.csv -e data.h5ad
        castor enrich results.csv -e expr.csv -t "Contextual Anomaly" --species mouse
        castor enrich results.csv -e data.h5ad --per-spot --top-spots 20
    """
    import numpy as np
    import pandas as pd

    from castor.enrichment.gene_analysis import identify_contributing_genes
    from castor.enrichment.pathway import run_pathway_enrichment
    from castor.preprocessing.normalize import normalize_expression

    console.print(BANNER)

    results = pd.read_csv(results_csv, index_col=0)

    # Fetch orthologs for mouse species
    orthologs = None
    if species.lower() == "mouse":
        from castor.enrichment.pathway import fetch_orthologs

        with _progress() as progress:
            task = progress.add_task("Fetching mouse→human orthologs...", total=None)
            try:
                orthologs = fetch_orthologs()
                progress.update(task, completed=1, total=1)
                console.print(
                    f"[green]Orthologs loaded ({len(orthologs):,} mappings)[/green]"
                )
            except Exception as exc:
                progress.update(task, completed=1, total=1)
                console.print(
                    f"[yellow]Ortholog fetch failed ({exc}); "
                    f"falling back to uppercase mapping[/yellow]"
                )

    # Load expression
    with _progress() as progress:
        task = progress.add_task("Loading expression data...", total=None)

        if expression.endswith(".h5ad"):
            from castor.io.loaders import load_anndata

            X, _, gene_names = load_anndata(expression)
        else:
            expr_df = pd.read_csv(expression, index_col=0)
            X = expr_df.values.astype(np.float64)
            gene_names = expr_df.columns.tolist()

        X_norm = normalize_expression(X)
        progress.update(task, completed=1, total=1)

    # Gene contributions (all genes for GSEA, display top_n)
    contrib_all = identify_contributing_genes(
        X_norm, results, gene_names=gene_names, diagnosis_type=diagnosis_type, top_n=None
    )

    if len(contrib_all) == 0:
        console.print(f"[yellow]No anomalies of type '{diagnosis_type}' found.[/yellow]")
        raise typer.Exit(0)

    contrib_display = contrib_all.head(top_n)

    # Display gene table
    gene_table = Table(
        title=f"Top {len(contrib_display)} Contributing Genes",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    gene_table.add_column("Rank", justify="right", style="dim")
    gene_table.add_column("Gene", style="cyan bold")
    gene_table.add_column("Score", justify="right", style="green")
    gene_table.add_column("Fold Change", justify="right", style="yellow")

    for i, row in contrib_display.iterrows():
        gene_table.add_row(
            str(i + 1), row["Gene"], f"{row['Score']:.4f}", f"{row['Fold_Change']:.3f}"
        )

    console.print()
    console.print(gene_table)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    contrib_display.to_csv(out / "contributing_genes.csv", index=False)

    # Gene contribution plot
    if not no_plot:
        from castor.visualization.plots import plot_contributing_genes

        try:
            plot_contributing_genes(contrib_display, save_path=str(out / "contributing_genes.png"),
                                   top_n=top_n)
        except Exception as exc:
            console.print(f"[yellow]Gene plot skipped: {exc}[/yellow]")

    # Pathway enrichment
    enrichment = None
    console.print()
    with _progress() as progress:
        task = progress.add_task("Running pathway enrichment...", total=None)
        try:
            enrichment = run_pathway_enrichment(
                contrib_all, species=species, database=database, orthologs=orthologs,
            )
            progress.update(task, completed=1, total=1)

            if len(enrichment) > 0:
                enrichment.to_csv(out / "enrichment_results.csv", index=False)
                console.print(
                    f"[green]Enrichment results saved ({len(enrichment)} pathways)[/green]"
                )
            else:
                console.print("[yellow]No significant pathways found.[/yellow]")
                enrichment = None
        except Exception as exc:
            progress.update(task, completed=1, total=1)
            console.print(f"[yellow]Enrichment skipped: {exc}[/yellow]")

    # Per-spot enrichment
    spot_enrichment = None
    if per_spot:
        from castor.enrichment.gene_analysis import identify_spot_contributing_genes

        console.print()
        with _progress() as progress:
            task = progress.add_task(
                f"Running per-spot enrichment (top {top_spots} spots)...", total=top_spots
            )
            spot_genes = identify_spot_contributing_genes(
                X_norm, results, gene_names=gene_names,
                top_n_spots=top_spots, top_n_genes=None,
            )

            # Save per-spot DEG tables
            deg_dir = out / "spot_deg"
            deg_dir.mkdir(parents=True, exist_ok=True)
            for spot_idx, genes_df in spot_genes.items():
                genes_df.head(top_n).to_csv(
                    deg_dir / f"spot_{spot_idx}_deg.csv", index=False
                )

            all_spot_enrichments: list[pd.DataFrame] = []
            for i, (spot_idx, genes_df) in enumerate(spot_genes.items()):
                try:
                    enr = run_pathway_enrichment(
                        genes_df, species=species, database=database,
                        orthologs=orthologs,
                    )
                    if len(enr) > 0:
                        enr.insert(0, "Spot_Index", spot_idx)
                        score = float(np.maximum(
                            results.iloc[spot_idx]["Local_Z"],
                            results.iloc[spot_idx]["Global_Z"],
                        ))
                        enr.insert(1, "Anomaly_Score", score)
                        all_spot_enrichments.append(enr)
                except Exception:
                    pass
                progress.update(task, completed=i + 1)

        if all_spot_enrichments:
            spot_enrichment = pd.concat(all_spot_enrichments, ignore_index=True)
            spot_enrichment.to_csv(out / "spot_enrichment_results.csv", index=False)
            n_spots_ok = spot_enrichment["Spot_Index"].nunique()
            n_paths = len(spot_enrichment)
            console.print(
                f"[green]Per-spot enrichment saved "
                f"({n_paths} pathways across {n_spots_ok} spots)[/green]"
            )
        else:
            console.print("[yellow]No significant per-spot enrichment found.[/yellow]")

    # Enrichment plots
    if not no_plot and enrichment is not None and len(enrichment) > 0:
        from castor.visualization.plots import plot_enrichment_bar, plot_enrichment_dotplot

        with _progress() as progress:
            task = progress.add_task("Generating enrichment figures...", total=None)
            try:
                plot_enrichment_bar(
                    enrichment, save_path=str(out / "enrichment_bar.png"), title=database,
                )
                plot_enrichment_dotplot(
                    enrichment, save_path=str(out / "enrichment_dotplot.png"), title=database,
                )
            except Exception as exc:
                console.print(f"[yellow]Enrichment plots skipped: {exc}[/yellow]")
            progress.update(task, completed=1, total=1)

    # Output summary
    tree = Tree(f"[bold green]{out}/[/bold green]", guide_style="dim")
    tree.add("[cyan]contributing_genes.csv[/cyan]")
    if (out / "contributing_genes.png").exists():
        tree.add("[cyan]contributing_genes.png[/cyan] -- gene bar chart")
    if (out / "enrichment_results.csv").exists():
        tree.add("[cyan]enrichment_results.csv[/cyan]")
    if (out / "enrichment_bar.png").exists():
        tree.add("[cyan]enrichment_bar.png[/cyan] -- pathway bar chart")
    if (out / "enrichment_dotplot.png").exists():
        tree.add("[cyan]enrichment_dotplot.png[/cyan] -- pathway dot plot")
    if (out / "spot_deg").exists():
        tree.add("[cyan]spot_deg/[/cyan] -- per-spot DEG tables")
    if (out / "spot_enrichment_results.csv").exists():
        tree.add("[cyan]spot_enrichment_results.csv[/cyan] -- per-spot enrichment")

    console.print()
    console.print(Panel(tree, title="Output Files", border_style="green"))


# -----------------------------------------------------------------------
# run  (all-in-one)
# -----------------------------------------------------------------------
@app.command()
def run(
    input_path: Annotated[str, typer.Argument(help="Path to .h5ad or expression CSV")],
    coords: Annotated[
        str | None,
        typer.Option("--coords", "-c", help="Path to coordinates CSV (required for CSV input)"),
    ] = None,
    output_dir: Annotated[
        str, typer.Option("--output-dir", "-o", help="Output directory")
    ] = "castor_output",
    species: Annotated[
        str, typer.Option("--species", help="Species: human or mouse")
    ] = "human",
    intrinsic: Annotated[
        str,
        typer.Option("--intrinsic", "-i", help="Intrinsic detection method"),
    ] = "pca_error",
    n_epochs: Annotated[int, typer.Option("--n-epochs", "-e", help="Training epochs")] = 100,
    device: Annotated[str, typer.Option("--device", "-d", help="Device: auto, cpu, cuda")] = "auto",
    per_spot: Annotated[
        bool, typer.Option("--per-spot", help="Run per-spot enrichment & DEG")
    ] = False,
    top_spots: Annotated[
        int, typer.Option("--top-spots", help="Number of top spots")
    ] = 10,
    database: Annotated[
        str, typer.Option("--database", help="Enrichr library name")
    ] = "GO_Biological_Process_2023",
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Minimal output")] = False,
) -> None:
    """[bold cyan]Run[/bold cyan] full pipeline: detect + enrich in one command.

    [bold]Examples:[/bold]
        castor run data.h5ad
        castor run data.h5ad --species mouse
        castor run data.h5ad --species mouse --per-spot
    """
    import numpy as np
    import pandas as pd

    from castor.api import CASTOR
    from castor.config import CASTORConfig
    from castor.enrichment.gene_analysis import identify_contributing_genes
    from castor.enrichment.pathway import run_pathway_enrichment
    from castor.io.exporters import save_parameters, save_results
    from castor.preprocessing.normalize import normalize_expression

    if not quiet:
        console.print(BANNER)

    logging.basicConfig(
        level=logging.INFO if not quiet else logging.WARNING,
        format="%(message)s",
    )

    cfg = CASTORConfig(
        n_epochs=n_epochs,
        intrinsic_method=intrinsic,
        device=device,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Detection ---
    inp = Path(input_path)
    if inp.suffix == ".csv" and coords is None:
        console.print(Panel("[red]--coords is required for CSV input[/red]", border_style="red"))
        raise typer.Exit(1)

    with _progress() as progress:
        task = progress.add_task("Running CASTOR detection...", total=None)
        castor = CASTOR(config=cfg)
        results = castor.fit_predict(str(inp), coords=coords, verbose=not quiet)
        progress.update(task, completed=1, total=1)

    save_results(results, out / "results.csv")
    save_parameters(castor.params_, out / "parameters.json")

    try:
        with _progress() as progress:
            task = progress.add_task("Generating plots...", total=None)
            castor.plot_results(save_path=str(out / "castor_6panel.png"))
            progress.update(task, completed=1, total=1)
    except Exception as exc:
        console.print(f"[yellow]Warning: could not generate plots: {exc}[/yellow]")

    # Summary table
    summary = Table(
        title="Detection Results", box=box.DOUBLE_EDGE,
        show_header=True, header_style="bold cyan",
    )
    summary.add_column("Diagnosis", style="bold")
    summary.add_column("Count", justify="right", style="green")
    summary.add_column("Percentage", justify="right", style="yellow")
    total = len(results)
    styles = {
        "Normal": "dim", "Contextual Anomaly": "blue",
        "Intrinsic Anomaly": "magenta", "Confirmed Anomaly": "red bold",
    }
    for diag in ("Normal", "Contextual Anomaly", "Intrinsic Anomaly", "Confirmed Anomaly"):
        count = (results["Diagnosis"] == diag).sum()
        pct = count / total * 100
        s = styles.get(diag, "")
        summary.add_row(f"[{s}]{diag}[/{s}]", f"{count:,}", f"{pct:.1f}%")
    console.print()
    console.print(summary)

    # --- Enrichment ---
    console.print()

    # Fetch orthologs for mouse
    orthologs = None
    if species.lower() == "mouse":
        from castor.enrichment.pathway import fetch_orthologs

        with _progress() as progress:
            task = progress.add_task("Fetching mouse→human orthologs...", total=None)
            try:
                orthologs = fetch_orthologs()
                progress.update(task, completed=1, total=1)
                console.print(f"[green]Orthologs loaded ({len(orthologs):,} mappings)[/green]")
            except Exception as exc:
                progress.update(task, completed=1, total=1)
                console.print(f"[yellow]Ortholog fetch failed ({exc}); uppercase fallback[/yellow]")

    # Load expression for enrichment
    with _progress() as progress:
        task = progress.add_task("Loading expression for enrichment...", total=None)
        if inp.suffix == ".h5ad":
            from castor.io.loaders import load_anndata
            X, _, gene_names = load_anndata(str(inp))
        else:
            expr_df = pd.read_csv(str(inp), index_col=0)
            X = expr_df.values.astype(np.float64)
            gene_names = expr_df.columns.tolist()
        X_norm = normalize_expression(X)
        progress.update(task, completed=1, total=1)

    # Gene contributions (all genes for GSEA)
    contrib_all = identify_contributing_genes(
        X_norm, results, gene_names=gene_names,
        diagnosis_type="Confirmed Anomaly", top_n=None,
    )

    enrich_out = out / "enrichment"
    enrich_out.mkdir(parents=True, exist_ok=True)

    if len(contrib_all) > 0:
        contrib_display = contrib_all.head(20)

        gene_table = Table(
            title=f"Top {len(contrib_display)} Contributing Genes",
            box=box.ROUNDED, header_style="bold magenta",
        )
        gene_table.add_column("Rank", justify="right", style="dim")
        gene_table.add_column("Gene", style="cyan bold")
        gene_table.add_column("Score", justify="right", style="green")
        gene_table.add_column("Fold Change", justify="right", style="yellow")
        for i, row in contrib_display.iterrows():
            gene_table.add_row(
                str(i + 1), row["Gene"], f"{row['Score']:.4f}", f"{row['Fold_Change']:.3f}"
            )
        console.print()
        console.print(gene_table)

        contrib_display.to_csv(enrich_out / "contributing_genes.csv", index=False)

        # Pathway enrichment with all genes
        with _progress() as progress:
            task = progress.add_task("Running pathway enrichment...", total=None)
            try:
                enrichment = run_pathway_enrichment(
                    contrib_all, species=species, database=database, orthologs=orthologs,
                )
                progress.update(task, completed=1, total=1)
                if len(enrichment) > 0:
                    enrichment.to_csv(enrich_out / "enrichment_results.csv", index=False)
                    console.print(
                        f"[green]Enrichment saved ({len(enrichment)} pathways)[/green]"
                    )

                    from castor.visualization.plots import (
                        plot_enrichment_bar,
                        plot_enrichment_dotplot,
                    )

                    plot_enrichment_bar(
                        enrichment, save_path=str(enrich_out / "enrichment_bar.png"),
                        title=database,
                    )
                    plot_enrichment_dotplot(
                        enrichment, save_path=str(enrich_out / "enrichment_dotplot.png"),
                        title=database,
                    )
                else:
                    console.print("[yellow]No significant pathways found.[/yellow]")
            except Exception as exc:
                progress.update(task, completed=1, total=1)
                console.print(f"[yellow]Enrichment skipped: {exc}[/yellow]")
    else:
        console.print("[yellow]No Confirmed Anomalies found; skipping enrichment.[/yellow]")

    # --- Per-spot DEG + enrichment ---
    if per_spot:
        from castor.enrichment.gene_analysis import identify_spot_contributing_genes

        console.print()
        with _progress() as progress:
            task = progress.add_task(
                f"Per-spot DEG + enrichment (top {top_spots} spots)...", total=top_spots,
            )

            spot_genes = identify_spot_contributing_genes(
                X_norm, results, gene_names=gene_names,
                top_n_spots=top_spots, top_n_genes=None,
            )

            deg_dir = enrich_out / "spot_deg"
            deg_dir.mkdir(parents=True, exist_ok=True)
            for spot_idx, genes_df in spot_genes.items():
                genes_df.head(20).to_csv(deg_dir / f"spot_{spot_idx}_deg.csv", index=False)

            all_spot_enrichments: list[pd.DataFrame] = []
            for i, (spot_idx, genes_df) in enumerate(spot_genes.items()):
                try:
                    enr = run_pathway_enrichment(
                        genes_df, species=species, database=database, orthologs=orthologs,
                    )
                    if len(enr) > 0:
                        enr.insert(0, "Spot_Index", spot_idx)
                        score = float(np.maximum(
                            results.iloc[spot_idx]["Local_Z"],
                            results.iloc[spot_idx]["Global_Z"],
                        ))
                        enr.insert(1, "Anomaly_Score", score)
                        all_spot_enrichments.append(enr)
                except Exception:
                    pass
                progress.update(task, completed=i + 1)

        if all_spot_enrichments:
            spot_enrichment = pd.concat(all_spot_enrichments, ignore_index=True)
            spot_enrichment.to_csv(enrich_out / "spot_enrichment_results.csv", index=False)
            n_spots_ok = spot_enrichment["Spot_Index"].nunique()
            console.print(
                f"[green]Per-spot: {len(spot_enrichment)} pathways "
                f"across {n_spots_ok} spots[/green]"
            )
        else:
            console.print("[yellow]No significant per-spot enrichment found.[/yellow]")

    # Output tree
    tree = Tree(f"[bold green]{out}/[/bold green]", guide_style="dim")
    tree.add("[cyan]results.csv[/cyan] -- detection results")
    tree.add("[cyan]parameters.json[/cyan] -- parameters")
    if (out / "castor_6panel.png").exists():
        tree.add("[cyan]castor_6panel.png[/cyan] -- 6-panel figure")
    enrich_tree = tree.add("[cyan]enrichment/[/cyan]")
    for f in sorted(enrich_out.iterdir()):
        if f.is_file():
            enrich_tree.add(f"[cyan]{f.name}[/cyan]")
        elif f.is_dir():
            sub = enrich_tree.add(f"[cyan]{f.name}/[/cyan]")
            for sf in sorted(f.iterdir()):
                sub.add(f"[dim]{sf.name}[/dim]")

    console.print()
    console.print(Panel(tree, title="Output Files", border_style="green"))


# -----------------------------------------------------------------------
# explain
# -----------------------------------------------------------------------
@app.command()
def explain(
    input_path: Annotated[str, typer.Argument(help="Path to .h5ad or expression CSV")],
    coords: Annotated[
        str | None,
        typer.Option("--coords", "-c", help="Path to coordinates CSV (required for CSV input)"),
    ] = None,
    output_dir: Annotated[
        str, typer.Option("--output-dir", "-o", help="Output directory")
    ] = "castor_explain",
    spot_indices: Annotated[
        str | None,
        typer.Option("--spots", "-s", help="Comma-separated spot indices to explain (e.g., '100,200,300')"),
    ] = None,
    top_n_spots: Annotated[
        int, typer.Option("--top-spots", help="Number of top anomaly spots to explain")
    ] = 10,
    top_n_genes: Annotated[
        int, typer.Option("--top-genes", help="Number of top genes per spot")
    ] = 15,
    diagnosis_type: Annotated[
        str,
        typer.Option("--diagnosis", "-t", help="Anomaly type to explain"),
    ] = "Contextual Anomaly",
    exclude_lncrna: Annotated[
        bool,
        typer.Option("--exclude-lncrna", "--no-lncrna", help="Exclude lncRNA-like genes (Gm*, *Rik, AL*, AC*, LINC*)"),
    ] = False,
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Attribution method: gradient or integrated_gradients"),
    ] = "gradient",
    n_steps: Annotated[
        int,
        typer.Option("--n-steps", help="Number of steps for Integrated Gradients (default: 50)"),
    ] = 50,
    n_epochs: Annotated[int, typer.Option("--n-epochs", "-e", help="Training epochs")] = 100,
    device: Annotated[str, typer.Option("--device", "-d", help="Device: auto, cpu, cuda")] = "auto",
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Minimal output")] = False,
) -> None:
    """[bold cyan]Explain[/bold cyan] which genes drive position prediction for anomalous spots.

    Uses gradient-based attribution to identify genes that contribute most
    to the model's position prediction error.

    [bold]Examples:[/bold]
        castor explain data.h5ad
        castor explain data.h5ad --spots 100,200,300
        castor explain data.h5ad --top-spots 20 --top-genes 20
        castor explain data.h5ad --exclude-lncrna  # Focus on protein-coding genes
        castor explain data.h5ad --method integrated_gradients  # Mathematically rigorous attribution
    """
    import pandas as pd

    from castor.api import CASTOR
    from castor.config import CASTORConfig
    from castor.io.exporters import save_parameters, save_results

    if not quiet:
        console.print(BANNER)

    logging.basicConfig(
        level=logging.INFO if not quiet else logging.WARNING,
        format="%(message)s",
    )

    cfg = CASTORConfig(
        n_epochs=n_epochs,
        device=device,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load + run
    inp = Path(input_path)
    if inp.suffix == ".csv" and coords is None:
        console.print(Panel("[red]--coords is required for CSV input[/red]", border_style="red"))
        raise typer.Exit(1)

    with _progress() as progress:
        task = progress.add_task("Running CASTOR pipeline...", total=None)
        castor = CASTOR(config=cfg)
        results = castor.fit_predict(str(inp), coords=coords, verbose=not quiet)
        progress.update(task, completed=1, total=1)

    # Parse spot indices
    spots_to_explain = None
    if spot_indices:
        spots_to_explain = [int(s.strip()) for s in spot_indices.split(",")]
        console.print(f"[cyan]Explaining spots: {spots_to_explain}[/cyan]")

    # Run explanation
    with _progress() as progress:
        task = progress.add_task("Computing gene attributions...", total=None)
        explanation = castor.explain_spots(
            spot_indices=spots_to_explain,
            top_n_spots=top_n_spots,
            top_n_genes=top_n_genes,
            diagnosis_type=diagnosis_type,
            exclude_lncrna=exclude_lncrna,
            method=method,
            n_steps=n_steps,
        )
        progress.update(task, completed=1, total=1)

    if exclude_lncrna:
        console.print("[cyan]lncRNA-like genes excluded (Gm*, *Rik, AL*, AC*, LINC*)[/cyan]")
    if method == "integrated_gradients":
        console.print(f"[cyan]Using Integrated Gradients (n_steps={n_steps})[/cyan]")

    if len(explanation["spot_indices"]) == 0:
        console.print(f"[yellow]No spots found for diagnosis type '{diagnosis_type}'[/yellow]")
        raise typer.Exit(0)

    # Save gene summary
    explanation["gene_summary"].to_csv(out / "gene_attribution_summary.csv", index=False)

    # Display top genes table
    is_ig = method == "integrated_gradients"
    col_name = "Mean |Attribution|" if is_ig else "Mean |Gradient|"
    col_key = "mean_abs_attribution" if is_ig else "mean_abs_gradient"

    gene_table = Table(
        title="Top Contributing Genes (across all spots)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    gene_table.add_column("Rank", justify="right", style="dim")
    gene_table.add_column("Gene", style="cyan bold")
    gene_table.add_column(col_name, justify="right", style="green")

    for i, row in explanation["gene_summary"].head(20).iterrows():
        gene_table.add_row(
            str(i + 1), row["gene"], f"{row[col_key]:.6f}"
        )

    console.print()
    console.print(gene_table)

    # Save and display per-spot details
    spot_details_dir = out / "spot_details"
    spot_details_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    for rank, detail in enumerate(explanation["spot_details"], 1):
        spot_idx = detail.attrs["spot_idx"]
        local_z = detail.attrs["local_z"]
        disp_x = detail.attrs["displacement_x"]
        disp_y = detail.attrs["displacement_y"]

        # Save CSV with rank in filename
        detail.to_csv(spot_details_dir / f"rank{rank}_spot{spot_idx}_genes.csv", index=False)

        # Display table
        title = f"Rank #{rank}: Spot {spot_idx} (Local_Z={local_z:.2f}, Displacement=({disp_x:.4f}, {disp_y:.4f}))"
        if is_ig and "sum_attribution_x" in detail.attrs:
            # Show completeness check for IG
            err_x = detail.attrs.get("completeness_error_x", 0)
            err_y = detail.attrs.get("completeness_error_y", 0)
            title += f"\n  [dim]Completeness error: ({err_x:.4f}, {err_y:.4f})[/dim]"

        spot_table = Table(
            title=title,
            box=box.SIMPLE,
            header_style="bold cyan",
        )
        spot_table.add_column("Gene", style="cyan")
        spot_table.add_column("Attribution" if is_ig else "Gradient", justify="right", style="green")
        spot_table.add_column("Expression", justify="right", style="yellow")

        attr_col = "attribution_mag" if is_ig else "gradient"
        for _, row in detail.head(10).iterrows():
            spot_table.add_row(
                row["gene"], f"{row[attr_col]:+.6f}", f"{row['expression']:.4f}"
            )

        console.print(spot_table)
        console.print()

    # Generate plots for top spots
    with _progress() as progress:
        task = progress.add_task("Generating explanation figures...", total=len(explanation["spot_indices"]))
        for i, spot_idx in enumerate(explanation["spot_indices"][:5]):  # Max 5 plots
            rank = i + 1
            try:
                castor.plot_spot_explanation(
                    spot_idx,
                    save_path=str(out / f"rank{rank}_spot{spot_idx}_explanation.png"),
                    top_n_genes=top_n_genes,
                    rank=rank,
                    diagnosis_type=diagnosis_type,
                    exclude_lncrna=exclude_lncrna,
                    method=method,
                    n_steps=n_steps,
                )
            except Exception as exc:
                console.print(f"[yellow]Plot for spot {spot_idx} skipped: {exc}[/yellow]")
            progress.update(task, completed=i + 1)

    # Output tree
    tree = Tree(f"[bold green]{out}/[/bold green]", guide_style="dim")
    tree.add("[cyan]gene_attribution_summary.csv[/cyan] -- overall gene ranking")
    details_tree = tree.add("[cyan]spot_details/[/cyan] -- per-spot gene attributions")
    for f in sorted(spot_details_dir.iterdir()):
        details_tree.add(f"[dim]{f.name}[/dim]")

    plots = list(out.glob("spot_*_explanation.png"))
    if plots:
        for p in plots[:5]:
            tree.add(f"[cyan]{p.name}[/cyan] -- spot explanation figure")

    console.print()
    console.print(Panel(tree, title="Output Files", border_style="green"))


# -----------------------------------------------------------------------
# info
# -----------------------------------------------------------------------
@app.command()
def info(
    methods: Annotated[
        bool,
        typer.Option("--methods", "-m", help="List registered intrinsic detectors"),
    ] = False,
) -> None:
    """[bold cyan]Show[/bold cyan] CASTOR version and system information.

    [bold]Examples:[/bold]
        castor info
        castor info --methods
    """
    from castor import __version__

    console.print(BANNER)

    # Version table
    tbl = Table(box=box.ROUNDED, show_header=False)
    tbl.add_column("", style="cyan")
    tbl.add_column("")

    tbl.add_row("Version", f"[bold]{__version__}[/bold]")
    tbl.add_row("Package", "castor-st")

    import torch

    tbl.add_row("PyTorch", torch.__version__)
    tbl.add_row(
        "CUDA available",
        "[green]Yes[/green]" if torch.cuda.is_available() else "[red]No[/red]",
    )
    if torch.cuda.is_available():
        tbl.add_row("CUDA device", torch.cuda.get_device_name(0))

    console.print(tbl)

    if methods:
        from castor.detection.registry import IntrinsicDetectorRegistry

        console.print()
        mtbl = Table(
            title="Registered Intrinsic Detectors",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        mtbl.add_column("Method", style="cyan bold")
        mtbl.add_column("Description", style="dim")

        descs = {
            "pca_error": "PCA reconstruction error (default)",
            "lof": "Local Outlier Factor",
            "isolation_forest": "Isolation Forest",
            "mahalanobis": "Mahalanobis distance via Elliptic Envelope",
            "ocsvm": "One-Class SVM",
        }

        for name in IntrinsicDetectorRegistry.list_methods():
            mtbl.add_row(name, descs.get(name, "(user-registered)"))

        console.print(mtbl)

        console.print()
        console.print(
            Panel(
                "[dim]Register custom detectors with:[/dim]\n\n"
                "[bold]from castor import register_intrinsic_detector[/bold]\n\n"
                "[bold]@register_intrinsic_detector('my_method')[/bold]\n"
                "def my_detector(X, **kwargs):\n"
                "    ...",
                title="Custom Detectors",
                border_style="blue",
            )
        )
