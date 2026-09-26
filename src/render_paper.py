"""Render the project manuscript Markdown to a self-contained print HTML file."""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def inline(text: str) -> str:
    value = html.escape(text, quote=False)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        value,
    )
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    return value


def cells(line: str) -> list[str]:
    return [item.strip() for item in line.strip().strip("|").split("|")]


def is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line.strip()))


def render(markdown: str) -> str:
    lines = markdown.splitlines()
    chunks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            chunks.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        if line.lstrip().startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            parsed = [cells(row) for row in table_lines if not is_table_separator(row)]
            if parsed:
                head, *body = parsed
                header = "".join(f"<th>{inline(cell)}</th>" for cell in head)
                rows = "".join(
                    "<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>"
                    for row in body
                )
                chunks.append(f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>")
            continue

        ordered = re.match(r"^\s*\d+\.\s+(.+)$", line)
        unordered = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if ordered or unordered:
            is_ordered = ordered is not None
            tag = "ol" if is_ordered else "ul"
            items: list[str] = []
            while i < len(lines):
                match = re.match(r"^\s*\d+\.\s+(.+)$", lines[i]) if is_ordered else re.match(r"^\s*[-*+]\s+(.+)$", lines[i])
                if not match:
                    break
                items.append(f"<li>{inline(match.group(1))}</li>")
                i += 1
            chunks.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue

        paragraph: list[str] = []
        while i < len(lines) and lines[i].strip():
            if paragraph and (
                re.match(r"^#{1,6}\s+", lines[i])
                or lines[i].lstrip().startswith("|")
                or re.match(r"^\s*(\d+\.|[-*+])\s+", lines[i])
            ):
                break
            raw = lines[i]
            hard_break = raw.endswith("  ")
            paragraph.append(inline(raw.rstrip()))
            if hard_break:
                paragraph.append("<br>")
            elif i + 1 < len(lines) and lines[i + 1].strip():
                paragraph.append(" ")
            i += 1
        content = "".join(paragraph).strip()
        if content:
            style = ' class="status"' if content.startswith("<strong>Status:</strong>") else ""
            chunks.append(f"<p{style}>{content}</p>")

    return "\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("paper_draft.md"))
    parser.add_argument("--output", type=Path, default=Path("paper_draft.html"))
    args = parser.parse_args()

    body = render(args.input.read_text(encoding="utf-8"))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training-Time Image Augmentation on MobileNetV3-Small</title>
<style>
@page {{ size: A4; margin: 18mm 17mm 19mm; }}
* {{ box-sizing: border-box; }}
body {{ color: #171717; font-family: Georgia, 'Times New Roman', serif; font-size: 10pt; line-height: 1.42; }}
h1 {{ font-size: 20pt; line-height: 1.18; text-align: center; margin: 0 0 15pt; }}
h1 + p {{ text-align: center; line-height: 1.35; margin: 0 auto 3pt; }}
h2 {{ font-size: 13pt; margin: 19pt 0 7pt; page-break-after: avoid; }}
h3 {{ font-size: 11pt; margin: 13pt 0 5pt; page-break-after: avoid; }}
p {{ margin: 0 0 8pt; text-align: justify; orphans: 3; widows: 3; }}
p.status {{ text-align: center; font-size: 9pt; color: #444; margin: 7pt 0 14pt; }}
ul, ol {{ margin: 2pt 0 9pt 20pt; padding-left: 14pt; }}
li {{ padding-left: 2pt; margin: 0 0 3pt; }}
table {{ width: 100%; border-collapse: collapse; margin: 7pt 0 11pt; font-size: 7.4pt; line-height: 1.2; table-layout: auto; page-break-inside: auto; }}
thead {{ display: table-header-group; }}
tr {{ page-break-inside: avoid; page-break-after: auto; }}
th, td {{ border: 0.5pt solid #888; padding: 3pt 4pt; vertical-align: top; overflow-wrap: anywhere; }}
th {{ background: #f0f2f4; text-align: left; }}
code {{ font-family: 'Courier New', monospace; font-size: 8pt; overflow-wrap: anywhere; }}
a {{ color: #174a7e; overflow-wrap: anywhere; }}
@media print {{ a {{ color: #174a7e; }} }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    args.output.write_text(document, encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
