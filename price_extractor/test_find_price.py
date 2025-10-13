import re, os
import pdfplumber

PDF_DIR = "./USDA_price_report"
files = sorted([os.path.join(PDF_DIR,f) for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])
pdf = files[-1]
with pdfplumber.open(pdf) as doc:
    text = '\n'.join(p.extract_text() or '' for p in doc.pages)

# replicate functions from extract_price.py

def clean_num(s):
    if s is None:
        return None
    s = str(s)
    s = s.replace(',', '').replace('$', '').strip()
    try:
        return float(s)
    except:
        return None


def find_price_singleline(text, pattern, take='avg'):
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return {'value': None, 'low': None, 'high': None}
    line_start = text.rfind('\n', 0, m.start())
    line_end = text.find('\n', m.end())
    if line_start == -1: line_start = 0
    if line_end == -1: line_end = len(text)
    line = text[line_start:line_end]

    m_avg = re.search(r'(?:avg|average|wtd(?:\s*avg)?)\s*[:\-]?\s*\$?\s*([\d,]+\.?\d*)', line, flags=re.I)
    if m_avg:
        v = clean_num(m_avg.group(1))
        return {'value': v, 'low': None, 'high': None}

    m_rng = re.search(r'\$?\s*([\d,]+\.?\d*)\s*(?:to|\-|\u2013|\u2014|–|—)\s*\$?\s*([\d,]+\.?\d*)', line, flags=re.I)
    if m_rng:
        low = clean_num(m_rng.group(1))
        high = clean_num(m_rng.group(2))
        if take == 'avg' and low is not None and high is not None:
            return {'value': (low+high)/2.0, 'low': low, 'high': high}
        return {'value': None, 'low': low, 'high': high}

    m_any = re.search(r'\$?\s*([\d,]+\.?\d*)', line)
    if m_any:
        return {'value': clean_num(m_any.group(1)), 'low': None, 'high': None}

    return {'value': None, 'low': None, 'high': None}

patterns = {
    'classII': r'Class\s*II\s*Milk',
    'repl': r'Replacement\s*[-–]\s*Fresh\s*Cow',
    'bull1': r'Calves?.*0\s*[-–]?\s*14\s*days.*Bulls?(?:.*No\.?\s*1|.*#\s*1)?',
    'bull2': r'Calves?.*0\s*[-–]?\s*14\s*days.*Bulls?(?:.*No\.?\s*2|.*#\s*2)?',
    'heif1': r'Calves?.*0\s*[-–]?\s*14\s*days.*Heifers?(?:.*No\.?\s*1|.*#\s*1)?',
    'heif2': r'Calves?.*0\s*[-–]?\s*14\s*days.*Heifers?(?:.*No\.?\s*2|.*#\s*2)?',
}

for name, pat in patterns.items():
    res = find_price_singleline(text, pat)
    print(name, res)

# Also test with a narrower text around the class II line
m = re.search(r'Class II Milk.*Replacement - Fresh Cow', text, flags=re.I)
if m:
    start = max(0, m.start()-100)
    end = min(len(text), m.end()+100)
    print('\nCLASS II CONTEXT:\n', text[start:end].replace('\n','\\n'))

print('\nDone')
