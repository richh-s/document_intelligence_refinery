import sys
from pathlib import Path
from docling.document_converter import DocumentConverter

pdf_path = Path("data/tax_expenditure_ethiopia_2021_22.pdf")
print(f"Parsing {pdf_path}")
converter = DocumentConverter()
res = converter.convert(pdf_path)
doc = res.document

count = 0
for node, data in doc.iterate_items():
    print(f"Node: {node}, Type: {type(data)}, HasLabel: {hasattr(data, 'label')}")
    if hasattr(data, 'label'):
        print(f"  Label: {data.label}, HasProv: {hasattr(data, 'prov')}, Prov: {getattr(data, 'prov', None)}")
    count += 1
    if count > 20:
        break
print(f"Total items inspected: {count}")
