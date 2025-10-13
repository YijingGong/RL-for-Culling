import re, sys, os
import pdfplumber

PDF_DIR = "./USDA_price_report"
files = sorted([os.path.join(PDF_DIR,f) for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])
if not files:
    print('No pdfs')
    sys.exit(1)

# Patterns to inspect
PAT_CLASS_II = re.compile(r'Class\s*II\s*Milk', re.I)
PAT_REPL_FRESH = re.compile(r'Replacement\s*[-–]\s*Fresh\s*Cow', re.I)
PAT_CALF = re.compile(r'Calves?.*0\s*[-–]?\s*14\s*days', re.I)
PAT_CULL_HEADER = re.compile(r'Negotiated\s*Direct\s*Cow/Bull\s*Price', re.I)

# Choose a couple of files to inspect (last two)
for pdf in files[-2:]:
    print('\n---', os.path.basename(pdf), '---')
    with pdfplumber.open(pdf) as doc:
        text = "\n".join(p.extract_text() or '' for p in doc.pages)

    # show first 400 chars
    print('\nFIRST 400 CHARS:\n', text[:400])

    for pat_name, pat in [('CLASS_II', PAT_CLASS_II), ('REPL_FRESH', PAT_REPL_FRESH), ('CALF', PAT_CALF), ('CULL', PAT_CULL_HEADER)]:
        print(f"\n== Searching for {pat_name} ==")
        for m in pat.finditer(text):
            start = max(0, m.start()-120)
            end = min(len(text), m.end()+120)
            snippet = text[start:end]
            # normalize newlines
            snippet = '\n'.join(line.strip() for line in snippet.splitlines())
            print('\nMATCH:\n', snippet)

    # Also show some numeric frequency to see which numbers are common
    nums = re.findall(r'\$?\s*([0-9]{1,4}(?:\,[0-9]{3})*(?:\.\d+)?)', text)
    nums_count = {}
    for n in nums:
        nums_count[n] = nums_count.get(n,0)+1
    top = sorted(nums_count.items(), key=lambda x: -x[1])[:10]
    print('\nTop numbers found in text (value:count):')
    for v,c in top:
        print(v,':',c)
