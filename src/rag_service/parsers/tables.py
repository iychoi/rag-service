"""Split rendered tables (markdown or HTML) into a repeatable header and body rows."""

import re

from rag_service.parsers.base import TableParts

_MD_SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
_HTML_ROW = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)


def markdown_table_parts(text: str) -> TableParts | None:
    """GFM table → header row + separator as `head`, one body row per entry. None if not a table."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2 or not _MD_SEPARATOR.match(lines[1]):
        return None
    return TableParts(head=lines[:2], rows=lines[2:])


def html_table_parts(text: str) -> TableParts | None:
    """HTML table → `<tr>` rows; a first row with `<th>` cells is the repeated head.

    thead/tbody wrappers and table attributes are dropped; the pieces stay readable HTML.
    """
    rows = _HTML_ROW.findall(text)
    if not rows:
        return None
    head = [rows[0]] if "<th" in rows[0].lower() else []
    return TableParts(head=head, rows=rows[len(head) :], open="<table>", close="</table>", sep="")


def table_parts(text: str) -> TableParts | None:
    if text.lstrip().lower().startswith("<table"):
        return html_table_parts(text)
    return markdown_table_parts(text)


def markdown_table(rows: list[list[str]], header_rows: int) -> TableParts | None:
    """Render a cell grid as a GFM table. The first `header_rows` rows (at least one, since GFM
    needs a header) become the repeated head."""
    if not rows:
        return None
    width = max(len(r) for r in rows)
    if width == 0:
        return None

    def line(cells: list[str]) -> str:
        cells = [c.replace("|", "\\|").replace("\n", " ").strip() for c in cells]
        cells += [""] * (width - len(cells))
        return "| " + " | ".join(cells) + " |"

    n_head = min(max(header_rows, 1), len(rows))
    head = [line(rows[0]), "| " + " | ".join(["---"] * width) + " |"]
    head += [line(r) for r in rows[1:n_head]]
    return TableParts(head=head, rows=[line(r) for r in rows[n_head:]])
