import argparse
from pathlib import Path

def extract_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using pdfplumber if available, else fall back to pypdf."""
    text = []
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
    except Exception:
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                text.append(page.extract_text() or "")
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}")
    return "\n\n".join(text)


def extract_code_blocks(text: str) -> str:
    """A light heuristic to extract Python-like code blocks from the document text."""
    lines = text.splitlines()
    code_lines = []
    buffer = []

    def flush():
        nonlocal buffer
        if buffer:
            code_lines.append("\n".join(buffer))
            buffer = []

    # Keywords that often indicate Python code
    starters = (
        "import ",
        "from ",
        "def ",
        "class ",
        "if ",
        "for ",
        "while ",
        "try:",
        "except ",
        "with ",
        "plt.",
        "pd.",
        "np.",
        "tk.",
        "ttk.",
    )

    in_block = False
    for raw in lines:
        line = raw.rstrip()
        # treat triple backticks or code indicators as explicit block boundaries
        if line.strip().startswith("```"):
            if in_block:
                flush()
                in_block = False
            else:
                in_block = True
            continue

        if in_block:
            buffer.append(line)
            continue

        # heuristic: lines that look like code
        if (line.startswith(starters) or line.endswith(":") or ("(" in line and ")" in line and ":" in line)):
            in_block = True
            buffer.append(line)
        else:
            # paragraph lines end a potential block
            flush()

    flush()
    # join blocks with two newlines
    return "\n\n".join(block for block in code_lines if block.strip())


def main():
    parser = argparse.ArgumentParser(description="Extract text and Python-like code blocks from a PDF")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the input PDF file")
    parser.add_argument("--out", type=str, required=True, help="Path to save extracted text")
    parser.add_argument("--code", type=str, required=True, help="Path to save extracted code blocks")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_text = Path(args.out)
    out_code = Path(args.code)

    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_code.parent.mkdir(parents=True, exist_ok=True)

    text = extract_text(pdf_path)
    out_text.write_text(text, encoding="utf-8")

    code = extract_code_blocks(text)
    if code.strip():
        out_code.write_text(code, encoding="utf-8")
    else:
        out_code.write_text("# No identifiable code blocks found in the document.\n", encoding="utf-8")

    print(f"Text saved to: {out_text}")
    print(f"Code saved to: {out_code}")


if __name__ == "__main__":
    main()