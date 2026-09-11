import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from docx import Document

doc = Document(r'c:\Users\rahma\Documents\Playground\LLM-Project\DMUD_MSC_Thesis-Final_v1.3.docx')

body = doc._body._element
print("=== APPENDIX SEQUENCE OF ELEMENTS ===")
in_app = False
for idx, child in enumerate(body):
    tag = child.tag.split('}')[-1]
    if tag == 'p':
        text = child.text if child.text else ""
        for t in child.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
            text += t.text if t.text else ""
        text = text.strip()
        if 'APPENDICES' in text:
            in_app = True
        if in_app and text:
            print(f"Elem #{idx} (P): {text[:90]}")
    elif tag == 'tbl':
        if in_app:
            # find which table it is
            for t_i, tbl in enumerate(doc.tables):
                if tbl._element is child:
                    header = [c.text.strip().replace('\n', ' ') for c in tbl.rows[0].cells]
                    print(f"Elem #{idx} (TBL #{t_i+1}): {header[:3]}")
