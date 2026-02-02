#!/usr/bin/env python
"""
CASTOR Experiments CLI

Unified CLI for running experiments and generating figures.

Usage:
    python cli.py exp all           # Run all experiments
    python cli.py exp run 1 3 5     # Run experiments 1, 3, 5
    python cli.py exp list          # List all experiments
    python cli.py fig all           # Generate all figures
    python cli.py fig one 1         # Generate figure 1
    python cli.py fig one S1        # Generate figure S1
    python cli.py fig one 1 -p a    # Generate figure 1 panel a only
    python cli.py fig list          # List all figures
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

# Setup paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "experiments" / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Console
console = Console()

# Apps
app = typer.Typer(
    name="castor-exp",
    help="🧬 CASTOR Inverse Prediction Experiments CLI",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

exp_app = typer.Typer(
    help="🧪 Run experiments",
    no_args_is_help=True,
)

fig_app = typer.Typer(
    help="📊 Generate figures",
    no_args_is_help=True,
)

app.add_typer(exp_app, name="exp")
app.add_typer(fig_app, name="fig")

# =============================================================================
# Experiment Registry
# =============================================================================

EXPERIMENTS = {
    1: ("Cross-Detection AUC", "exp01_cross_detection", "exp01_cross-detection_auc.csv"),
    2: ("Competitor Comparison", "exp02_competitor", "exp02_competitor_comparison.csv"),
    3: ("Position Accuracy", "exp03_position_accuracy", None),
    4: ("Noise Robustness", "exp04_noise_robustness", None),
    5: ("Independence Analysis", "exp05_independence", "exp05_independence_analysis.csv"),
    7: ("Ablation Studies", "exp07_ablation", None),
    8: ("Clean Training (Legacy)", "exp08_clean_training", None),
    9: ("Clean Training Fixed", "exp09_clean_training_fixed", None),
    10: ("Multi-Scenario Validation", "exp10_multi_scenario", None),
    11: ("Real Data Validation", "exp11_real_data", None),
    12: ("Embedding Comparison", "exp12_embedding_comparison", None),
    13: ("Scalability Analysis", "exp13_scalability", "exp13_scalability.csv"),
    14: ("HER2ST Validation", "exp14_her2st_validation", "exp14_her2st_validation.csv"),
    15: ("Full Benchmark", "exp15_full_benchmark", "exp15_full_benchmark.csv"),
    16: ("Gene Contribution", "exp16_real_data_gene_analysis", "exp16_gene_analysis_summary.csv"),
    17: ("HER2ST Transplantation", "exp17_her2st_transplantation", "exp17_transplantation.csv"),
    18: ("Interpretability", "exp18_interpretability", "exp18_interpretability.csv"),
    19: ("Clustered Ectopic", "exp19_clustered_ectopic", "exp19_clustered_ectopic.csv"),
    20: ("Region Transplantation", "exp20_region_transplantation", "exp20_region_transplantation.csv"),
}

# =============================================================================
# Figure Registry
# =============================================================================

FIGURES = {
    "1": ("Concept", "fig1", ["a", "b", "c", "d"]),
    "2": ("Selectivity", "fig2", ["a", "b", "c"]),
    "3": ("Robustness", "fig3", ["a", "b", "c"]),
    "4": ("Independence", "fig4", ["a", "b", "c"]),
    "5": ("HER2ST Validation", "fig5", ["a", "b", "c"]),
    "6": ("Transplantation", "fig6", ["a", "b", "c", "d"]),
    "7": ("Gene Analysis", "fig7", ["a", "b", "c", "d"]),
    "S1": ("Noise Robustness", "figS1", ["a", "b", "c"]),
    "S2": ("Ablation Study", "figS2", ["a", "b", "c", "d"]),
    "S3": ("Interpretability", "figS3", ["a", "b", "c", "d"]),
    "S4": ("Statistics", "figS4", ["a", "b", "c", "d"]),
    "S5": ("Scalability", "figS5", ["a", "b", "c"]),
    "S6": ("Synthetic Data", "figS6", ["a", "b", "c", "d"]),
    "8": ("Clustered Advantage", "fig8", ["a", "b", "c"]),
    "9": ("Region Transplantation", "fig9", ["a", "b", "c", "d"]),
}

# =============================================================================
# Experiment Commands
# =============================================================================

@exp_app.command("list")
def exp_list():
    """📋 List all available experiments."""
    table = Table(
        title="🧪 Available Experiments",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("ID", style="bold", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Module")
    table.add_column("Status", justify="center")

    for exp_id, (name, module, output_file) in sorted(EXPERIMENTS.items()):
        # Check if results exist
        if output_file:
            result_path = RESULTS_DIR / output_file
        else:
            result_path = RESULTS_DIR / f"exp{exp_id:02d}_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"

        status = "✅" if result_path.exists() else "⬜"
        table.add_row(str(exp_id), name, module, status)

    console.print(table)


@exp_app.command("run")
def exp_run(
    ids: List[int] = typer.Argument(..., help="Experiment IDs to run"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode (3 seeds)"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-Q", help="Verbose output"),
):
    """▶️ Run specific experiments."""
    from config import DEFAULT_CONFIG, QUICK_CONFIG
    config = QUICK_CONFIG if quick else DEFAULT_CONFIG

    console.print(Panel(
        f"[bold]Running experiments: {ids}[/bold]\n"
        f"Mode: {'Quick (3 seeds)' if quick else 'Full (30 seeds)'}\n"
        f"Device: {config.device}",
        title="🚀 Experiment Runner",
        border_style="blue",
    ))

    results = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running experiments...", total=len(ids))

        for exp_id in ids:
            if exp_id not in EXPERIMENTS:
                console.print(f"[red]Unknown experiment: {exp_id}[/red]")
                progress.advance(task)
                continue

            name, module_name, output_file = EXPERIMENTS[exp_id]
            progress.update(task, description=f"Exp {exp_id}: {name}")

            try:
                module = _import_experiment(module_name)
                if module is None:
                    console.print(f"[yellow]Skipping exp{exp_id}: module not found[/yellow]")
                    progress.advance(task)
                    continue

                result = module.run(config, verbose=verbose)
                results[exp_id] = result

                # Save results
                if output_file:
                    out_path = RESULTS_DIR / output_file
                else:
                    out_path = RESULTS_DIR / f"exp{exp_id:02d}_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"

                if len(result) > 0:
                    result.to_csv(out_path, index=False)
                    console.print(f"[green]✓ Exp {exp_id} saved to {out_path.name}[/green]")
                else:
                    console.print(f"[yellow]⚠ Exp {exp_id}: No results[/yellow]")

            except Exception as e:
                console.print(f"[red]✗ Exp {exp_id} failed: {e}[/red]")

            progress.advance(task)

    console.print(f"\n[bold green]Completed {len(results)}/{len(ids)} experiments[/bold green]")


@exp_app.command("all")
def exp_all(
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode (3 seeds)"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-Q", help="Verbose output"),
):
    """🚀 Run all experiments."""
    exp_run(list(EXPERIMENTS.keys()), quick=quick, verbose=verbose)


def _import_experiment(module_name: str):
    """Import an experiment module."""
    try:
        import importlib
        return importlib.import_module(f"experiments.{module_name}")
    except ImportError:
        return None


# =============================================================================
# Figure Commands
# =============================================================================

@fig_app.command("list")
def fig_list():
    """📋 List all available figures."""
    table = Table(
        title="📊 Available Figures",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("ID", style="bold", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Panels")
    table.add_column("Status", justify="center")

    for fig_id, (name, module, panels) in FIGURES.items():
        # Check if combined figure exists
        combined_path = FIGURES_DIR / module / "combined.pdf"
        status = "✅" if combined_path.exists() else "⬜"
        panel_str = ", ".join(panels)
        table.add_row(fig_id, name, panel_str, status)

    console.print(table)


@fig_app.command("one")
def fig_one(
    fig_id: str = typer.Argument(..., help="Figure ID (1-7 or S1-S6)"),
    panel: Optional[List[str]] = typer.Option(None, "--panel", "-p", help="Specific panels (a, b, c, ...)"),
):
    """🎨 Generate a specific figure."""
    fig_id = fig_id.upper() if fig_id.lower().startswith('s') else fig_id

    if fig_id not in FIGURES:
        console.print(f"[red]Unknown figure: {fig_id}[/red]")
        console.print(f"Available figures: {', '.join(FIGURES.keys())}")
        raise typer.Exit(1)

    name, module_name, available_panels = FIGURES[fig_id]

    console.print(Panel(
        f"[bold]Generating Figure {fig_id}: {name}[/bold]\n"
        f"Panels: {', '.join(panel) if panel else 'all'}",
        title="🎨 Figure Generator",
        border_style="green",
    ))

    try:
        module = _import_figure_module(module_name)
        if module is None:
            console.print(f"[red]Figure module not found: {module_name}[/red]")
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if panel:
                # Generate specific panels
                for p in panel:
                    if p.lower() not in available_panels:
                        console.print(f"[yellow]Unknown panel: {p}. Available: {available_panels}[/yellow]")
                        continue

                    task = progress.add_task(f"Panel {p}...", total=None)
                    panel_module = _import_panel_module(module_name, p.lower())
                    if panel_module and hasattr(panel_module, 'main'):
                        panel_module.main()
                        console.print(f"[green]✓ Panel {p} complete[/green]")
                    else:
                        console.print(f"[yellow]⚠ Panel {p} not found[/yellow]")
                    progress.remove_task(task)
            else:
                # Generate combined figure
                task = progress.add_task("Generating figure...", total=None)
                if hasattr(module, 'combined') and hasattr(module.combined, 'main'):
                    module.combined.main()
                elif hasattr(module, 'main'):
                    module.main()
                progress.remove_task(task)
                console.print(f"[green]✓ Figure {fig_id} complete[/green]")

    except Exception as e:
        console.print(f"[red]Error generating figure: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@fig_app.command("main")
def fig_main():
    """🖼️ Generate all main figures (1-7)."""
    main_figs = [k for k in FIGURES.keys() if not k.startswith('S')]
    _generate_multiple_figures(main_figs, "Main Figures")


@fig_app.command("supp")
def fig_supp():
    """📑 Generate all supplementary figures (S1-S6)."""
    supp_figs = [k for k in FIGURES.keys() if k.startswith('S')]
    _generate_multiple_figures(supp_figs, "Supplementary Figures")


@fig_app.command("all")
def fig_all():
    """🖼️ Generate all figures."""
    _generate_multiple_figures(list(FIGURES.keys()), "All Figures")


def _generate_multiple_figures(fig_ids: List[str], title: str):
    """Generate multiple figures with progress."""
    console.print(Panel(
        f"[bold]Generating {title}[/bold]\n"
        f"Figures: {', '.join(fig_ids)}",
        title="🎨 Figure Generator",
        border_style="green",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating figures...", total=len(fig_ids))

        for fig_id in fig_ids:
            name, module_name, _ = FIGURES[fig_id]
            progress.update(task, description=f"Fig {fig_id}: {name}")

            try:
                module = _import_figure_module(module_name)
                if module is None:
                    console.print(f"[yellow]⚠ Fig {fig_id}: module not found[/yellow]")
                    progress.advance(task)
                    continue

                if hasattr(module, 'combined') and hasattr(module.combined, 'main'):
                    module.combined.main()
                elif hasattr(module, 'main'):
                    module.main()

                console.print(f"[green]✓ Fig {fig_id}[/green]")

            except Exception as e:
                console.print(f"[red]✗ Fig {fig_id}: {e}[/red]")

            progress.advance(task)

    console.print(f"\n[bold green]Figure generation complete[/bold green]")


def _import_figure_module(module_name: str):
    """Import a figure module."""
    try:
        import importlib
        sys.path.insert(0, str(SCRIPT_DIR / "experiments" / "visualization"))
        return importlib.import_module(module_name)
    except ImportError as e:
        console.print(f"[dim]Import error: {e}[/dim]")
        return None


def _import_panel_module(fig_module: str, panel: str):
    """Import a specific panel module."""
    try:
        import importlib
        sys.path.insert(0, str(SCRIPT_DIR / "experiments" / "visualization"))
        return importlib.import_module(f"{fig_module}.panel_{panel}")
    except ImportError:
        return None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    app()
