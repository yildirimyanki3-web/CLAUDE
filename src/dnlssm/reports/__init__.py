"""Numeric-only automated reporting: Markdown reports plus CSV/LaTeX table export.

:func:`~dnlssm.reports.report_builder.build_experiment_report` assembles a
full Markdown report from whichever pipeline stages actually ran (data
acquisition, preprocessing, model selection, optimization, diagnostics,
robustness), stating plainly what did not run or did not converge rather
than omitting it. :func:`~dnlssm.reports.table_export.export_table` writes
any result table to both CSV and manuscript-ready LaTeX in one call.
"""

from dnlssm.reports.report_builder import ReportContext, build_experiment_report
from dnlssm.reports.table_export import export_table

__all__ = ["ReportContext", "build_experiment_report", "export_table"]
