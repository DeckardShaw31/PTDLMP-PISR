from typing import List, Dict, Any
import pandas as pd

def format_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Formats a clean GitHub-flavored markdown table."""
    col_widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        s_row = [str(x) for x in row]
        for i, val in enumerate(s_row):
            if len(val) > col_widths[i]:
                col_widths[i] = len(val)
        str_rows.append(s_row)

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = ["| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in str_rows]

    return "\n".join([header_line, sep_line] + data_lines)

def format_latex_table(headers: List[str], rows: List[List[Any]], caption: str, label: str) -> str:
    """Formats a publication-ready Booktabs LaTeX table."""
    num_cols = len(headers)
    alignments = "l" + "r" * (num_cols - 1)
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignments}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule"
    ]
    
    for row in rows:
        s_row = [str(x).replace("%", "\\%").replace("_", "\\_") for x in row]
        latex_lines.append(" & ".join(s_row) + " \\\\")
        
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_lines)
