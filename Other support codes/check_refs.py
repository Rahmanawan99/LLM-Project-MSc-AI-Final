import argparse
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from docx import Document

parser = argparse.ArgumentParser(description="Check table and figure references in a Word document")
parser.add_argument("document", help="Path to the .docx document")
args = parser.parse_args()

doc = Document(args.document)

print("=== CHECKING TABLE AND FIGURE REFERENCES IN BODY ===")
for i, p in enumerate(doc.paragraphs):
    if i > 95 and i < 438:
        t = p.text.strip()
        if any(w in t.lower() for w in ['table', 'figure', 'appendix', 'appendices']):
            print(f"[{i}] {t[:100]}")
