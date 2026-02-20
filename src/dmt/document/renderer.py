"""Render a populated DocumentBuilder to Markdown.

This is the minimal viable renderer — proof that DMT can produce
structured scientific documents from decorator-accumulated sections.
"""

from pathlib import Path
from collections import OrderedDict

import pandas as pd


def render_markdown(title: str, sections: OrderedDict, output_dir: str | Path) -> Path:
    """Render sections to a Markdown file.

    Parameters
    ----------
    title : str
        Document title.
    sections : OrderedDict
        label -> dict with keys: narrative (str), data (DataFrame or None),
        illustration (str path or None).
    output_dir : path
        Directory to write the report into.

    Returns the path to the generated .md file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"# {title}\n"]

    for label, content in sections.items():
        section_name = content.get("name", label.replace("_", " ").title())
        lines.append(f"\n## {section_name}\n")

        if content.get("narrative"):
            lines.append(content["narrative"])
            lines.append("")

        if content.get("data") is not None:
            df = content["data"]
            if isinstance(df, pd.DataFrame):
                lines.append(df.to_markdown(index=False))
                lines.append("")
                # Also save CSV
                csv_path = output_dir / f"{label}.csv"
                df.to_csv(csv_path, index=False)

        if content.get("illustration"):
            ill = content["illustration"]
            lines.append(f"![{section_name}]({ill})")
            lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path
