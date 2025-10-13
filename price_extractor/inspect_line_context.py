import re, os
import pdfplumber

PDF_DIR = "./USDA_price_report"
files = sorted([os.path.join(PDF_DIR,f) for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])
if not files:
    print('No pdfs')
    raise SystemExit(1)

patterns = {
    'CLASS_II': re.compile(r'Class\s*II\s*Milk', re.I),
    'REPL_FRESH': re.compile(r'Replacement\s*[-–]\s*Fresh\s*Cow', re.I),
    'CALF_BULL_NO1': re.compile(r'Calves?.*0\s*[-–]?\s*14\s*days.*Bulls?(?:.*No\.?\s*1|.*#\s*1)?', re.I),
    'CALF_BULL_NO2': re.compile(r'Calves?.*0\s*[-–]?\s*14\s*days.*Bulls?(?:.*No\.?\s*2|.*#\s*2)?', re.I),
    'CALF_HEIF_NO1': re.compile(r'Calves?.*0\s*[-–]?\s*14\s*days.*Heifers?(?:.*No\.?\s*1|.*#\s*1)?', re.I),
    'CALF_HEIF_NO2': re.compile(r'Calves?.*0\s*[-–]?\s*14\s*days.*Heifers?(?:.*No\.?\s*2|.*#\s*2)?', re.I),
}

# pick one PDF to inspect (last)
pdf = files[-1]
print('Inspecting', os.path.basename(pdf))
with pdfplumber.open(pdf) as doc:
    text = '\n'.join(p.extract_text() or '' for p in doc.pages)

for name, pat in patterns.items():
    print('\n---', name, '---')
    m = pat.search(text)
    if not m:
        print('No match')
        continue
    # replicate find_price_singleline line extraction
    line_start = text.rfind('\n', 0, m.start())
    line_end = text.find('\n', m.end())
    if line_start == -1: line_start = 0
    if line_end == -1: line_end = len(text)
    line = text[line_start:line_end]
    print('LINE RAW:\n', line.replace('\n','\\n'))
    # numbers found on the line
    nums = re.findall(r'\$?\s*([0-9]{1,4}(?:\,[0-9]{3})*(?:\.\d+)?)', line)
    print('Numbers on line:', nums)
    # show 120 chars before and after match
    start = max(0, m.start()-120)
    end = min(len(text), m.end()+120)
    print('CONTEXT:\n', text[start:end].replace('\n','\\n'))

print('\nDone')
