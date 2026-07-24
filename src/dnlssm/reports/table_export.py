"""CSV and LaTeX table export, suitable for direct inclusion in an academic manuscript."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_FLOAT_FORMAT = "%.4f"


def export_table(df: pd.DataFrame, csv_path: Path, tex_path: Path, caption: str | None = None) -> tuple[Path, Path]:
    """Writes ``df`` to both ``csv_path`` and ``tex_path`` (booktabs-style LaTeX)."""
    csv_path, tex_path = Path(csv_path), Path(tex_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)

    latex_body = df.to_latex(index=False, float_format=_FLOAT_FORMAT, na_rep="--", escape=True)
    if caption:
        latex_body = (
            "\\begin{table}[htbp]\n\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"{latex_body}"
            "\\end{table}\n"
        )
    tex_path.write_text(latex_body, encoding="utf-8")
    return csv_path, tex_path


__all__ = ["export_table"]
