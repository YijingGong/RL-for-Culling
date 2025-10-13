# parse_usda_2957_pdfs.py
import re
import glob
import os
from datetime import datetime

try:
    import pdfplumber
    import pandas as pd
except Exception as e:
    # defer import error until runtime; helpful message printed when script executed
    pdfplumber = None
    pd = None

PDF_DIR = "./USDA_price_report"

# ---------- helpers ----------
def clean_num(s):
    if s is None:
        return None
    s = str(s)
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except:
        return None

def extract_first_date(text, fallback_name=None):
    # Try formats like "October 6, 2025"
    m = re.search(r'([A-Z][a-z]+ \d{1,2}, \d{4})', text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%B %d, %Y").date()
        except:
            pass
    # Try ISO-like "2025-10-06"
    m = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        except:
            pass
    # Try filename fallback
    if fallback_name:
        m = re.search(r'(20\d{2})[-_](\d{2})[-_](\d{2})', fallback_name)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
    return None

def extract_block(text, header_regex, stop_regex=r"\n\s*\n"):
    """
    Return the block of text starting at a header (regex) until a blank line (or stop_regex).
    """
    m = re.search(header_regex, text, flags=re.I)
    if not m:
        return None
    start = m.start()
    # from start to next blank line or end
    block = text[start:]
    stop = re.search(stop_regex, block, flags=re.I)
    if stop:
        block = block[:stop.start()]
    return block

def find_price_singleline(text, pattern, take="avg", unit_whitelist=None):
    """
    Find a single line by regex pattern and pull best price:
    - supports 'avg' value if present in line,
    - else range "low - high" (returns mid if take='avg', or returns low/high separately if needed)
    - else first number on the line.
    Returns dict with keys: value, low, high
    """
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return {"value": None, "low": None, "high": None}
    # Focus on a short window immediately after the match to avoid grabbing
    # unrelated numbers that may appear later on the same visual line
    start_idx = m.start()
    # capture up to N chars after the match (covers typical table rows)
    N = 240
    line = text[start_idx:start_idx + N]
    # If that window is empty for some reason, fall back to the original line logic
    if not line.strip():
        line_start = text.rfind("\n", 0, m.start())
        line_end = text.find("\n", m.end())
        if line_start == -1: line_start = 0
        if line_end == -1: line_end = len(text)
        line = text[line_start:line_end]

    # Try to detect "Avg", "Average", "Wtd Avg"
    m_avg = re.search(r'(?:avg|average|wtd(?:\s*avg)?)\s*[:\-]?\s*\$?\s*([\d,]+\.?\d*)', line, flags=re.I)
    if m_avg:
        v = clean_num(m_avg.group(1))
        return {"value": v, "low": None, "high": None}

    # Prefer numbers that are followed by allowed units (e.g., "/cwt", "/hd").
    # If unit_whitelist is None, use a conservative default that prefers /cwt over others
    default_units = ['/cwt', '/hd', '/head', '/ton', '/bu', '/lb']
    units = unit_whitelist if unit_whitelist is not None else default_units
    for unit in units:
        # allow optional space before the unit and word boundary after
        u_pat = re.escape(unit)
        m_unit = re.search(r'\$?\s*([\d,]+\.?\d*)\s*' + u_pat + r'\b', line, flags=re.I)
        if m_unit:
            return {"value": clean_num(m_unit.group(1)), "low": None, "high": None}

    # Prefer explicit decimal numbers (prices often have decimals)
    # Choose the first decimal number that appears after the pattern match
    m_decimal = re.search(r'([\d]+\.[\d]+)', line)
    if m_decimal:
        # find the first decimal that occurs after the keyword match (m.start())
        decimals = [m.group(1) for m in re.finditer(r'([\d]+\.[\d]+)', line)]
        if decimals:
            return {"value": clean_num(decimals[0]), "low": None, "high": None}

    # Try range low-high but only accept ranges that look like prices (contain decimals)
    m_rng = re.search(r'\$?\s*([\d,]+\.?\d*)\s*(?:to|\-|\u2013|\u2014|–|—)\s*\$?\s*([\d,]+\.?\d*)', line, flags=re.I)
    if m_rng:
        g1 = m_rng.group(1)
        g2 = m_rng.group(2)
        # accept the range as price only if one of the numbers has a decimal or the range is followed by a unit
        after = line[m_rng.end():m_rng.end()+20]
        if ('.' in g1) or ('.' in g2) or re.search(r'(?:/cwt|/hd|/ton|/bu|/lb)', after, flags=re.I):
            low = clean_num(g1)
            high = clean_num(g2)
            if take == "avg" and low is not None and high is not None:
                return {"value": (low+high)/2.0, "low": low, "high": high}
            return {"value": None, "low": low, "high": high}

    # Fallback: first numeric token (prefer those with decimals if possible)
    m_any = re.search(r'\$?\s*([\d,]+\.?\d*)', line)
    if m_any:
        return {"value": clean_num(m_any.group(1)), "low": None, "high": None}

    return {"value": None, "low": None, "high": None}

# ---------- regex targets ----------
# 1) Class II milk
PAT_CLASS_II = r'Class\s*II\s*Milk'

# 2) Replacement – Fresh Cow (some reports include the word 'Approved', some don't)
PAT_REPL_FRESH = r'Replacement\s*[-–]\s*Fresh\s*Cow'

# 3–6) Calves age 0–14 days – Bulls/Heifers (No. 1 and No. 2 if present)
# Make the "No. 1" / "No 1" part optional and tolerant to spacing/characters
PAT_CALF_BULL_NO1 = r'Calves?\s*.*0\s*[-–]?\s*14\s*days.*Bulls?(?:.*No\.?\s*1|.*#\s*1)?'
PAT_CALF_BULL_NO2 = r'Calves?\s*.*0\s*[-–]?\s*14\s*days.*Bulls?(?:.*No\.?\s*2|.*#\s*2)?'
PAT_CALF_HEIF_NO1 = r'Calves?\s*.*0\s*[-–]?\s*14\s*days.*Heifers?(?:.*No\.?\s*1|.*#\s*1)?'
PAT_CALF_HEIF_NO2 = r'Calves?\s*.*0\s*[-–]?\s*14\s*days.*Heifers?(?:.*No\.?\s*2|.*#\s*2)?'

# 7) Block under Negotiated Direct Cow/Bull Price – All Breeds Dressed (Domestic)
# We grab the whole Domestic dressed block and then parse each line with a generic line parser.
# 7) Block under Negotiated Direct Cow/Bull Price – All Breeds Dressed (Domestic)
# Be permissive about parentheses and extra words around Domestic/Dressed
PAT_CULL_HEADER = r'Negotiated\s*Direct\s*Cow/Bull\s*Price\s*[-–]\s*All\s*Breeds\s*Dressed[\s:,-]*.*Domestic'

# Price capture helpers for the cull block
RE_PRICE_RANGE = re.compile(r'\$?\s*([\d,]+\.?\d*)\s*(?:to|\-|\u2013|\u2014|–|—)\s*\$?\s*([\d,]+\.?\d*)', re.I)
RE_PRICE_AVG   = re.compile(r'(?:avg|average|wtd(?:\s*avg)?)\s*[:\-]?\s*\$?\s*([\d,]+\.?\d*)', re.I)
RE_PRICE_ANY   = re.compile(r'\$?\s*([\d,]+\.?\d*)')

def parse_cull_block(block_text):
    """
    Parse all lines under the Domestic dressed header.
    Return list of dicts: {category, low, high, avg}
    """
    rows = []
    # Split into lines, skip the header line
    lines = block_text.splitlines()
    if lines:
        lines = lines[1:]
    for ln in lines:
        ln_clean = re.sub(r'\s+', ' ', ln).strip()
        if not ln_clean:
            continue
        # try to find numeric info
        low = high = avg = None

        m_avg = RE_PRICE_AVG.search(ln_clean)
        if m_avg:
            avg = clean_num(m_avg.group(1))

        m_rng = RE_PRICE_RANGE.search(ln_clean)
        if m_rng:
            low = clean_num(m_rng.group(1))
            high = clean_num(m_rng.group(2))

        # If nothing matched but a number exists, treat as "value" and place into avg
        if avg is None and low is None and high is None:
            m_any = RE_PRICE_ANY.search(ln_clean)
            if m_any:
                avg = clean_num(m_any.group(1))

        # Extract a category label (strip units and numbers)
        # Take text before first number as the category
        m_first_num = re.search(r'\$?\s*[\d,]+\.?\d*', ln_clean)
        cat = ln_clean
        if m_first_num:
            cat = ln_clean[:m_first_num.start()].strip(" -–—:")
        # Remove redundant header words if any linger
        cat = re.sub(r'^\s*(Domestic)\s*', '', cat, flags=re.I).strip()

        if any(v is not None for v in (low, high, avg)):
            rows.append({"category": cat, "price_low": low, "price_high": high, "price_avg": avg})
    return rows


def parse_cull_meat_prices(block_text):
    """
    Parse breaker/boner/cutter lines for the 500 lbs and up category.
    Return list of dicts: {category, lean, weight_class, price_low, price_high, price_avg}
    """
    rows = []
    lines = block_text.splitlines()
    for ln in lines:
        ln_clean = re.sub(r'\s+', ' ', ln).strip()
        if not ln_clean:
            continue
        # look for Breaker/Boner/Cutter
        m_cat = re.search(r'\b(Breaker|Boner|Cutter)\b', ln_clean, flags=re.I)
        if not m_cat:
            continue
        category = m_cat.group(1).title()

        # require 500 lbs and up (be permissive)
        if not re.search(r'500\s*lbs(?:\s*(?:and)?\s*up)?', ln_clean, flags=re.I):
            # skip non-500-lb lines
            continue

        # extract lean percent if present
        lean = None
        m_lean = re.search(r'(\d{1,3}%\s*Lean)', ln_clean, flags=re.I)
        if m_lean:
            lean = m_lean.group(1)

        # extract price avg or range
        low = high = avg = None
        m_avg = RE_PRICE_AVG.search(ln_clean)
        if m_avg:
            avg = clean_num(m_avg.group(1))

        m_rng = RE_PRICE_RANGE.search(ln_clean)
        if m_rng:
            low = clean_num(m_rng.group(1))
            high = clean_num(m_rng.group(2))

        # if only a single number present, treat as avg
        if avg is None and low is None and high is None:
            m_any = RE_PRICE_ANY.search(ln_clean)
            if m_any:
                avg = clean_num(m_any.group(1))

        # weight class normalization
        weight_class = '500 lbs and up'

        if any(v is not None for v in (low, high, avg)):
            rows.append({
                'category': category,
                'lean': lean,
                'weight_class': weight_class,
                'price_low': low,
                'price_high': high,
                'price_avg': avg,
            })
    return rows

# ---------- main extraction ----------
summary_rows = []
cull_rows = []

pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
if not pdf_files:
    raise SystemExit(f"No PDFs found in {PDF_DIR}")

for pdf_path in pdf_files:
    with pdfplumber.open(pdf_path) as doc:
        text = "\n".join(page.extract_text() or "" for page in doc.pages)

    rep_date = extract_first_date(text, os.path.basename(pdf_path))
    row = {
        "date": rep_date,
        "classII_milk": None,
        "replacement_fresh_cow": None,
        "calves_bulls_no1": None,
        "calves_bulls_no2": None,
        "calves_heifers_no1": None,
        "calves_heifers_no2": None,
        "source_pdf": os.path.basename(pdf_path),
    }

    # Class II milk: prefer per-cwt values
    r = find_price_singleline(text, PAT_CLASS_II, unit_whitelist=['/cwt'])
    row["classII_milk"] = r["value"] or r["low"]

    # Replacement – Fresh Cow
    # Replacement is typically per head (/hd); allow that
    r = find_price_singleline(text, PAT_REPL_FRESH, unit_whitelist=['/hd', '/head'])
    row["replacement_fresh_cow"] = r["value"] or r["low"]

    # Calves (Bulls/Heifers, No. 1 & 2)
    # Calves are shown per cwt
    r = find_price_singleline(text, PAT_CALF_BULL_NO1, unit_whitelist=['/cwt'])
    row["calves_bulls_no1"] = r["value"] or r["low"]

    r = find_price_singleline(text, PAT_CALF_BULL_NO2, unit_whitelist=['/cwt'])
    row["calves_bulls_no2"] = r["value"] or r["low"]

    r = find_price_singleline(text, PAT_CALF_HEIF_NO1, unit_whitelist=['/cwt'])
    row["calves_heifers_no1"] = r["value"] or r["low"]

    r = find_price_singleline(text, PAT_CALF_HEIF_NO2, unit_whitelist=['/cwt'])
    row["calves_heifers_no2"] = r["value"] or r["low"]

    summary_rows.append(row)

    # Cull dressed (Domestic) block
    block = extract_block(text, PAT_CULL_HEADER)
    if block:
        # parse generic cull items (feed ingredients)
        items = parse_cull_block(block)
        for it in items:
            it["date"] = rep_date
            it["source_pdf"] = os.path.basename(pdf_path)
            cull_rows.append(it)

        # parse meat prices for Breaker/Boner/Cutter (500 lbs and up)
        meat_items = parse_cull_meat_prices(block)
        for it in meat_items:
            it["date"] = rep_date
            it["source_pdf"] = os.path.basename(pdf_path)
            # keep separate list under a different key name so we can save a dedicated CSV
            if 'meat_rows' not in globals():
                meat_rows = []
            meat_rows.append(it)

# ---------- save ----------
summary_df = pd.DataFrame(summary_rows).sort_values("date")
summary_df.to_csv("usda_2957_summary.csv", index=False)

cull_df = pd.DataFrame(cull_rows).sort_values(["date", "category"])
cull_df.to_csv("usda_2957_cull_dressed_domestic.csv", index=False)

try:
    meat_rows
except NameError:
    meat_rows = []

meat_df = pd.DataFrame(meat_rows)
if not meat_df.empty:
    meat_df = meat_df.sort_values(["date", "category"])
    meat_df.to_csv("usda_2957_cull_meat_prices.csv", index=False)
    print("Saved:  usda_2957_cull_meat_prices.csv")

print("Saved:\n  usda_2957_summary.csv\n  usda_2957_cull_dressed_domestic.csv")
print("\nPreview summary:")
print(summary_df.tail(3))
print("\nPreview cull:")
print(cull_df.tail(5))
