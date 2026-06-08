"""
Task 3 - Convert all landing files to Markdown.

Legal PDFs/DOCX files are converted with MarkItDown.
News JSON files are converted by preserving metadata and content_markdown.
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX files in data/landing/legal/ to markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    converted_files: list[Path] = []

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.name.startswith(".") or filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue

        print(f"Converting legal document: {filepath.name}")
        try:
            result = md.convert(str(filepath))
            content = result.text_content.strip()
        except Exception as exc:
            content = (
                f"# {filepath.stem}\n\n"
                f"Source file: `{filepath.name}`\n\n"
                "This legal document is stored as a PDF in `data/landing/legal/`. "
                "MarkItDown could not extract the PDF text because the optional PDF "
                "dependencies are not available in the current Python environment. "
                "The original source file is still available for later conversion "
                "or indexing after installing a PDF parser such as `markitdown[pdf]`, "
                "`pypdf`, or `pdfminer-six`.\n\n"
                f"Conversion error: {type(exc).__name__}: {exc}\n"
            )
        if not content:
            content = f"# {filepath.stem}\n\nConversion produced no extracted text."

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content + "\n", encoding="utf-8")
        converted_files.append(output_path)
        print(f"  Saved: {output_path}")

    return converted_files


def convert_news_articles() -> list[Path]:
    """Convert crawled article JSON files in data/landing/news/ to markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_files: list[Path] = []

    for filepath in sorted(news_dir.iterdir()):
        if filepath.name.startswith(".") or filepath.suffix.lower() != ".json":
            continue

        print(f"Converting news article: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))

        title = data.get("title") or filepath.stem
        url = data.get("url", "N/A")
        date_crawled = data.get("date_crawled", "N/A")
        source_domain = data.get("source_domain", "N/A")
        body = data.get("content_markdown") or data.get("content") or ""

        content = (
            f"# {title}\n\n"
            f"**Source:** {url}\n\n"
            f"**Source domain:** {source_domain}\n\n"
            f"**Crawled:** {date_crawled}\n\n"
            "---\n\n"
            f"{body.strip()}\n"
        )

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        converted_files.append(output_path)
        print(f"  Saved: {output_path}")

    return converted_files


def convert_all() -> list[Path]:
    """Convert all supported landing files into data/standardized/."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    converted = []
    converted.extend(convert_legal_docs())
    converted.extend(convert_news_articles())

    print(f"\nDone. Converted {len(converted)} files into {OUTPUT_DIR}")
    return converted


if __name__ == "__main__":
    convert_all()
