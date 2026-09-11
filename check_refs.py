import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from docx import Document

doc = Document(r'c:\Users\rahma\Documents\Playground\LLM-Project\DMUD_MSC_Thesis-Final_v1.3.docx')

print("=== CHECKING TABLE AND FIGURE REFERENCES IN BODY ===")
for i, p in enumerate(doc.paragraphs):
    if i > 95 and i < 438:
        t = p.text.strip()
        if any(w in t.lower() for w in ['table', 'figure', 'appendix', 'appendices']):
            print(f"[{i}] {t[:100]}")
