"""Validate a stats parquet file against the canonical contract.

Thin CLI wrapper over :func:`stats_spec.validate_stats_parquet`. Checks the
column schema and metadata payload, and (unless ``--no-data-checks``) the
per-row value invariants.
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq
from rich import box
from rich.panel import Panel
from rich.table import Table

from ..console import console
from ..stats_spec import read_metadata, validate_stats_parquet


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add validate-stats specific arguments to the parser."""
    parser.add_argument("parquet_path", type=str, help="Path to the stats parquet file.")
    parser.add_argument(
        "--no-data-checks",
        action="store_true",
        help="Only check the schema and metadata (read the footer, not the rows).",
    )


def _summary_grid(path: str) -> Table:
    """Compact grid of the file's key parameters, shown on success."""
    meta = read_metadata(path)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="bold cyan")
    grid.add_column()
    grid.add_row("Source", f"📂  {meta.zarr_path}   var={meta.data_var} time={meta.time_var}")
    grid.add_row(
        "Datacube",
        f"🧊  {meta.time_depth} × {meta.width} × {meta.height}   stride {meta.step_t} × {meta.step_x} × {meta.step_y}",
    )
    grid.add_row("Range", f"📅  {meta.start_date:%Y-%m-%d} → {meta.end_date:%Y-%m-%d}")
    grid.add_row("Time step", f"🕒  {meta.time_step_minutes} min")
    grid.add_row("Max NaN", f"🔍  {meta.max_nan:,} per datacube")
    grid.add_row("Data kind", f"💧  {meta.data_kind}   (wet > {meta.wet_threshold:g} {meta.units or '?'})")
    grid.add_row("Rows", f"🔢  {pq.read_metadata(path).num_rows:,}")
    grid.add_row("Schema", f"📐  v{meta.schema_version}")
    return grid


def _fmt(value: object) -> str:
    """Compact cell formatting: grouped ints, 6-significant-figure floats."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _schema_table(path: str) -> Table:
    """Column-by-column view of the parquet schema (read from the footer)."""
    schema = pq.read_schema(path)
    table = Table(
        title="🧱 table structure", title_style="bold cyan", box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("column", style="bold")
    table.add_column("type")
    table.add_column("nullable")
    for i, field in enumerate(schema):
        table.add_row(str(i), field.name, str(field.type), "yes" if field.nullable else "no")
    return table


def _preview_table(path: str, n: int = 10) -> Table:
    """The first ``n`` rows, reading only the leading row group(s)."""
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    names = pf.schema_arrow.names
    table = Table(
        title=f"🔎 first {min(n, total)} rows",
        title_style="bold cyan",
        box=box.SIMPLE_HEAD,
        header_style="bold",
        pad_edge=False,
    )
    for name in names:
        table.add_column(name, justify="right")
    if total == 0:
        table.add_row(*["—"] * len(names))
        return table
    batch = next(pf.iter_batches(batch_size=n))
    cols = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
    for r in range(min(n, batch.num_rows)):
        table.add_row(*[_fmt(col[r]) for col in cols])
    return table


def run(args: argparse.Namespace) -> int:
    """Execute the validate-stats command."""
    path = args.parquet_path
    report = validate_stats_parquet(path, check_data=not args.no_data_checks)

    if report.ok and not report.warnings:
        console.print(
            Panel(
                _summary_grid(path),
                title="[bold green]✅ valid stats parquet[/]",
                subtitle=f"[dim]{path}[/]",
                border_style="green",
                expand=False,
            )
        )
    else:
        issues = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        issues.add_column("")
        issues.add_column("message")
        for w in report.warnings:
            issues.add_row("⚠️", w)
        for e in report.errors:
            issues.add_row("❌", e)

        if report.ok:
            title = f"[bold yellow]⚠️  valid — {len(report.warnings)} warning(s)[/]"
            border = "yellow"
        else:
            title = f"[bold red]❌ invalid — {len(report.errors)} violation(s)[/]"
            border = "red"
        console.print(Panel(issues, title=title, subtitle=f"[dim]{path}[/]", border_style=border, expand=False))

    # Structure and a row preview — useful for eyeballing a file, and helpful
    # for diagnosis even when the report flagged problems. Best-effort: a file
    # too broken to read here has already been reported above.
    try:
        console.print(_schema_table(path))
        if args.no_data_checks:
            console.print("[dim]row preview skipped (--no-data-checks)[/]")
        else:
            console.print(_preview_table(path))
    except Exception as exc:  # noqa: BLE001 - preview is non-essential
        console.print(f"[yellow]could not read table preview: {exc}[/]")

    return 0 if report.ok else 1
