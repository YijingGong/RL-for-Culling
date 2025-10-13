import re, glob, os
import pdfplumber

PDF_DIR = './USDA_price_report'
PAT = re.compile(r'Negotiated\s*Direct\s*Cow/Bull\s*Price\s*[-–]\s*All\s*Breeds\s*Dressed', re.I)

files = sorted(glob.glob(os.path.join(PDF_DIR, '*.pdf')))
for pdf in files:
    print('\n===', os.path.basename(pdf), '===')
    with pdfplumber.open(pdf) as doc:
        text = '\n'.join(p.extract_text() or '' for p in doc.pages)
    m = PAT.search(text)
    if not m:
        print('No cull header found')
        continue
    start = m.start()
    # capture 1000 chars after header to include table
    block = text[start:start+2000]
    print(block)
