"""
main.py
CLI entry point. Run: python main.py
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from data_gen import generate_synthetic_batch
from pipeline import run_batch_sync
from schemas import BatchRecoveryReport

console = Console(width=140)


def render_dashboard(results: list, report: BatchRecoveryReport) -> None:
    console.print(Panel.fit(
        f"[bold]Total Money At Risk:[/bold] ₹{report.total_money_at_risk_inr:,.2f}\n"
        f"[bold]Total Recovered:[/bold] ₹{report.total_recovered_inr:,.2f}\n"
        f"[bold]Recovery Rate:[/bold] {report.recovery_rate_pct}%\n"
        f"[bold]Escalations:[/bold] {report.escalations} / {report.total_transactions}\n"
        f"[bold]Hard Stops (non-recoverable):[/bold] {report.hard_stops}",
        title="[bold green]Revenue Recovery — Batch Summary[/bold green]",
    ))

    cause_table = Table(title="Root Cause Breakdown")
    cause_table.add_column("Root Cause")
    cause_table.add_column("Count", justify="right")
    for cause, count in sorted(report.root_cause_breakdown.items(), key=lambda x: -x[1]):
        cause_table.add_row(cause, str(count))
    console.print(cause_table)

    action_table = Table(title="Actions Taken")
    action_table.add_column("Action")
    action_table.add_column("Count", justify="right")
    for action, count in sorted(report.action_breakdown.items(), key=lambda x: -x[1]):
        action_table.add_row(action, str(count))
    console.print(action_table)


if __name__ == "__main__":
    with console.status("[bold cyan]Processing batch..."):
        transactions = generate_synthetic_batch(n=35)
        results, report = run_batch_sync(transactions)

    render_dashboard(results, report)