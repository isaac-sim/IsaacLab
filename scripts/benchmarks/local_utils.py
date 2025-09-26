
import math
import pandas as pd

###################
# Output Printing #
###################

def dataframe_to_markdown(
    df: pd.DataFrame, index: bool = True, floatfmt: str | None = ".3f", nan_rep: str = "", align: str = "left"
) -> str:
    """
    Convert a pandas DataFrame to a Markdown table (no extra deps).
    - index: include the index as the first column
    - floatfmt: e.g. '.2f' for floats; set None to use str() directly
    - nan_rep: string to show for NaNs/None
    - align: 'left' | 'center' | 'right'
    """

    def _fmt(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return nan_rep
        if isinstance(x, float) and floatfmt is not None:
            return format(x, floatfmt)
        return str(x)

    def _esc(s: str) -> str:
        # Escape pipes so they don't break the Markdown table
        return s.replace("|", r"\|")

    # Build header and rows
    headers = [str(c) for c in df.columns]
    rows = [[_fmt(v) for v in row] for row in df.to_numpy().tolist()]

    if index:
        headers = [df.index.name or ""] + headers
        idx_col = [str(i) for i in df.index.tolist()]
        rows = [[idx] + r for idx, r in zip(idx_col, rows)]

    # Compute column widths
    cols = list(zip(*([headers] + rows))) if headers else []
    widths = [max(len(_esc(h)), *(len(_esc(cell)) for cell in col)) for h, col in zip(headers, cols)]

    # Alignment rule
    def rule(w):
        if align == "right":
            return "-" * (w - 1) + ":"
        if align == "center":
            return ":" + "-" * (w - 2 if w > 2 else 1) + ":"
        return ":" + "-" * (w - 1)  # left

    # Build markdown lines
    def fmt_row(cells):
        return "| " + " | ".join(_esc(c).ljust(w) for c, w in zip(cells, widths)) + " |"

    header_line = fmt_row(headers)
    sep_line = "| " + " | ".join(rule(w) for w in widths) + " |"
    body_lines = [fmt_row(r) for r in rows]

    return "\n".join([header_line, sep_line, *body_lines])