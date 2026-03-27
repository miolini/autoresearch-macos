"""
SEC 10-K financial data pipeline for autoresearch.

Downloads 10-K filings from SEC EDGAR for financial companies (banks, insurance,
brokers), cleans HTML to plain text, packages into parquet shards, and trains
a domain-specific BPE tokenizer.

Usage:
    uv run prepare_10k.py              # full pipeline
    uv run prepare_10k.py --skip-download  # skip EDGAR download, just reprocess

Data is stored in ~/.cache/autoresearch-10k/.
"""

import os
import sys
import re
import json
import time
import math
import random
import pickle
import argparse
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pyarrow as pa
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

from sec_config import (
    CACHE_DIR_10K, DATA_DIR_10K, TOKENIZER_DIR_10K, RAW_DIR_10K,
    INDEX_PATH, SIC_CACHE_PATH,
    USER_AGENT, EDGAR_BASE, EDGAR_SUBMISSIONS,
    REQUEST_DELAY, MAX_RETRIES,
    FINANCIAL_SIC_CODES, FILING_YEARS, TARGET_NUM_FILINGS,
    MIN_FILING_TEXT_CHARS, MAX_FILING_TEXT_CHARS, MIN_ALPHA_RATIO, CHUNK_SIZE,
    VOCAB_SIZE, SPLIT_PATTERN, VAL_SHARD_NAME,
)

# Must match prepare.py
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

HEADERS = {"User-Agent": USER_AGENT}

# ---------------------------------------------------------------------------
# Stage A: Discover 10-K filings via EDGAR quarterly master index
# ---------------------------------------------------------------------------

def download_master_index(year, quarter):
    """Download and parse a quarterly master.idx file. Returns list of (cik, company, date, filename) for 10-K filings."""
    url = f"{EDGAR_BASE}/edgar/full-index/{year}/QTR{quarter}/master.idx"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  Warning: failed to download {url}: {e}")
                return []
            time.sleep(2 ** attempt)

    filings = []
    for line in resp.text.splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company, form_type, date_filed, filename = parts
        if form_type.strip() == "10-K":
            filings.append({
                "cik": cik.strip(),
                "company": company.strip(),
                "date": date_filed.strip(),
                "filename": filename.strip(),
            })
    return filings


def lookup_sic(cik, sic_cache):
    """Look up SIC code for a CIK via EDGAR submissions API. Returns int or None."""
    if cik in sic_cache:
        return sic_cache[cik]

    padded = cik.zfill(10)
    url = f"{EDGAR_SUBMISSIONS}/CIK{padded}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        sic = data.get("sic", "")
        sic_int = int(sic) if sic else None
    except Exception:
        sic_int = None

    sic_cache[cik] = sic_int
    return sic_int


def discover_filings():
    """Discover 10-K filings from financial companies. Returns list of filing dicts."""
    if os.path.exists(INDEX_PATH):
        print(f"Filing index: loading from {INDEX_PATH}")
        with open(INDEX_PATH) as f:
            filings = json.load(f)
        print(f"Filing index: {len(filings)} filings loaded")
        return filings

    os.makedirs(CACHE_DIR_10K, exist_ok=True)

    # Load SIC cache if exists
    sic_cache = {}
    if os.path.exists(SIC_CACHE_PATH):
        with open(SIC_CACHE_PATH) as f:
            sic_cache = json.load(f)

    # Step 1: Collect all 10-K filings from master indices
    print("Filing index: downloading quarterly master indices...")
    all_filings = []
    for year in FILING_YEARS:
        for quarter in range(1, 5):
            filings = download_master_index(year, quarter)
            all_filings.extend(filings)
            time.sleep(REQUEST_DELAY)
        print(f"  {year}: {len(all_filings)} total 10-K filings so far")

    print(f"Filing index: {len(all_filings)} total 10-K filings found across all years")

    # Step 2: Get unique CIKs and look up SIC codes
    unique_ciks = list(set(f["cik"] for f in all_filings))
    uncached = [c for c in unique_ciks if c not in sic_cache]
    print(f"Filing index: {len(unique_ciks)} unique CIKs, {len(uncached)} need SIC lookup")

    for i, cik in enumerate(uncached):
        lookup_sic(cik, sic_cache)
        if (i + 1) % 100 == 0:
            print(f"  SIC lookup: {i+1}/{len(uncached)}")
            # Save cache periodically
            with open(SIC_CACHE_PATH, "w") as f:
                json.dump(sic_cache, f)
        time.sleep(REQUEST_DELAY)

    # Save final SIC cache
    with open(SIC_CACHE_PATH, "w") as f:
        json.dump(sic_cache, f)

    # Step 3: Filter for financial companies
    financial_filings = []
    for filing in all_filings:
        sic = sic_cache.get(filing["cik"])
        if sic is not None and sic in FINANCIAL_SIC_CODES:
            financial_filings.append(filing)

    print(f"Filing index: {len(financial_filings)} filings from financial companies (SIC 6000-6411)")

    # Step 4: Sample if too many
    if len(financial_filings) > TARGET_NUM_FILINGS:
        random.seed(42)
        financial_filings = random.sample(financial_filings, TARGET_NUM_FILINGS)
        print(f"Filing index: sampled {TARGET_NUM_FILINGS} filings")

    # Save index
    with open(INDEX_PATH, "w") as f:
        json.dump(financial_filings, f, indent=2)
    print(f"Filing index: saved to {INDEX_PATH}")

    return financial_filings


# ---------------------------------------------------------------------------
# Stage B: Download filing documents
# ---------------------------------------------------------------------------

def filing_raw_path(filing):
    """Return local path for a raw filing."""
    accession = filing["filename"].split("/")[-1].replace(".txt", "")
    return os.path.join(RAW_DIR_10K, f"{filing['cik']}_{accession}.html")


def download_filing(filing):
    """Download a single 10-K filing document."""
    path = filing_raw_path(filing)
    if os.path.exists(path):
        return True

    # The filename in master.idx points to the filing wrapper .txt
    # We need to find the primary HTML document in the filing
    wrapper_url = f"https://www.sec.gov/Archives/{filing['filename']}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(wrapper_url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            with open(path, "w", encoding="utf-8", errors="replace") as f:
                f.write(resp.text)
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  Warning: failed to download {filing['company']} ({filing['cik']}): {e}")
                return False
            time.sleep(2 ** attempt)
    return False


def download_all_filings(filings):
    """Download all filing documents with rate limiting."""
    os.makedirs(RAW_DIR_10K, exist_ok=True)

    existing = sum(1 for f in filings if os.path.exists(filing_raw_path(f)))
    if existing == len(filings):
        print(f"Download: all {len(filings)} filings already downloaded")
        return

    print(f"Download: {existing}/{len(filings)} already downloaded, fetching rest...")
    to_download = [f for f in filings if not os.path.exists(filing_raw_path(f))]

    ok = existing
    for i, filing in enumerate(to_download):
        if download_filing(filing):
            ok += 1
        if (i + 1) % 50 == 0:
            print(f"  Download: {ok}/{len(filings)} complete")
        time.sleep(REQUEST_DELAY)

    print(f"Download: {ok}/{len(filings)} filings ready")


# ---------------------------------------------------------------------------
# Stage C: Clean HTML to plain text
# ---------------------------------------------------------------------------

class FilingHTMLParser(HTMLParser):
    """Extract text from SEC filing HTML, stripping tags and cleaning up."""

    SKIP_TAGS = {"script", "style", "xbrl", "ix:nonnumeric", "ix:nonfraction",
                 "ix:header", "ix:hidden", "ix:references"}
    BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                  "blockquote", "section", "article"}

    def __init__(self):
        super().__init__()
        self.result = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower in self.SKIP_TAGS:
            self.skip_depth += 1
        elif tag_lower in self.BLOCK_TAGS:
            self.result.append("\n")
        elif tag_lower == "td":
            self.result.append("\t")

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.result.append(data)

    def get_text(self):
        return "".join(self.result)


def extract_10k_document(raw_text):
    """Extract the 10-K document from an SGML filing wrapper."""
    # Try to find 10-K document section in SGML wrapper
    pattern = re.compile(
        r'<DOCUMENT>\s*<TYPE>10-K.*?<TEXT>(.*?)</TEXT>',
        re.DOTALL | re.IGNORECASE
    )
    match = pattern.search(raw_text)
    if match:
        return match.group(1)

    # If no SGML wrapper, the file itself is the document
    return raw_text


def clean_filing_text(raw_text):
    """Clean raw filing HTML/SGML into plain text."""
    # Extract the 10-K document from wrapper
    doc_text = extract_10k_document(raw_text)

    # Parse HTML
    parser = FilingHTMLParser()
    try:
        parser.feed(doc_text)
    except Exception:
        # Fallback: strip all tags with regex
        doc_text = re.sub(r'<[^>]+>', ' ', doc_text)
        return _postprocess(doc_text)

    return _postprocess(parser.get_text())


def _postprocess(text):
    """Clean up extracted text."""
    # Remove XBRL/XML artifacts that leak through
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove Unicode control characters (except newline, tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Collapse runs of whitespace on same line
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())
    # Remove common repeated headers
    text = re.sub(r'(?:Table of Contents\s*\n?){2,}', 'Table of Contents\n', text, flags=re.IGNORECASE)
    return text.strip()


def process_all_filings(filings):
    """Clean all downloaded filings. Returns list of (filing_dict, clean_text) pairs."""
    print("Cleaning: processing raw filings...")
    results = []
    skipped_missing = 0
    skipped_short = 0
    skipped_numeric = 0

    for i, filing in enumerate(filings):
        path = filing_raw_path(filing)
        if not os.path.exists(path):
            skipped_missing += 1
            continue

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        text = clean_filing_text(raw)

        # Quality filters
        if len(text) < MIN_FILING_TEXT_CHARS:
            skipped_short += 1
            continue

        alpha_count = sum(1 for c in text[:10000] if c.isalpha())
        if alpha_count / min(len(text), 10000) < MIN_ALPHA_RATIO:
            skipped_numeric += 1
            continue

        # Truncate very long filings
        if len(text) > MAX_FILING_TEXT_CHARS:
            text = text[:MAX_FILING_TEXT_CHARS]

        results.append((filing, text))

        if (i + 1) % 200 == 0:
            print(f"  Cleaning: {i+1}/{len(filings)} processed, {len(results)} kept")

    print(f"Cleaning: {len(results)} filings kept")
    print(f"  Skipped: {skipped_missing} missing, {skipped_short} too short, {skipped_numeric} too numeric")
    return results


# ---------------------------------------------------------------------------
# Stage D: Package into parquet shards
# ---------------------------------------------------------------------------

def chunk_text(text, target_size=CHUNK_SIZE):
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        if current_len + para_len + 2 > target_size and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def write_parquet_shards(cleaned_filings):
    """Split, shuffle, and write parquet shards."""
    os.makedirs(DATA_DIR_10K, exist_ok=True)

    print("Sharding: chunking filings...")
    all_chunks = []
    for filing, text in cleaned_filings:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    print(f"Sharding: {len(all_chunks)} chunks from {len(cleaned_filings)} filings")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(all_chunks)

    # Split 90/10 train/val
    val_size = max(1, len(all_chunks) // 10)
    val_chunks = all_chunks[:val_size]
    train_chunks = all_chunks[val_size:]

    print(f"Sharding: {len(train_chunks)} train, {len(val_chunks)} val chunks")

    # Write val shard with the magic name that prepare.py expects
    val_table = pa.table({"text": val_chunks})
    val_path = os.path.join(DATA_DIR_10K, VAL_SHARD_NAME)
    pq.write_table(val_table, val_path, row_group_size=1024)
    print(f"Sharding: wrote val shard to {val_path}")

    # Write train shards (~50K rows each to get a few shards)
    rows_per_shard = 50_000
    num_shards = max(1, math.ceil(len(train_chunks) / rows_per_shard))

    for shard_idx in range(num_shards):
        start = shard_idx * rows_per_shard
        end = min(start + rows_per_shard, len(train_chunks))
        shard_data = train_chunks[start:end]
        table = pa.table({"text": shard_data})
        shard_path = os.path.join(DATA_DIR_10K, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, shard_path, row_group_size=1024)
        print(f"Sharding: wrote train shard {shard_idx} ({len(shard_data)} rows)")

    print(f"Sharding: {num_shards} train shards + 1 val shard written to {DATA_DIR_10K}")


# ---------------------------------------------------------------------------
# Stage E: Train domain-specific BPE tokenizer
# ---------------------------------------------------------------------------

def text_iterator_10k(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training shards (excluding val shard)."""
    files = sorted(
        f for f in os.listdir(DATA_DIR_10K)
        if f.endswith(".parquet") and f != VAL_SHARD_NAME
    )
    nchars = 0
    for filename in files:
        filepath = os.path.join(DATA_DIR_10K, filename)
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer_10k():
    """Train BPE tokenizer on 10-K text. Mirrors prepare.py logic exactly."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR_10K, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR_10K, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR_10K}")
        return

    os.makedirs(TOKENIZER_DIR_10K, exist_ok=True)

    print("Tokenizer: training BPE tokenizer on 10-K text...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator_10k(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe_10k",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # Build token_bytes lookup for BPB evaluation
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "The allowance for loan losses increased to $2.3 billion in Q4 2024."
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SEC 10-K data for autoresearch")
    parser.add_argument("--skip-download", action="store_true", help="Skip EDGAR download, just reprocess existing files")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR_10K}")
    print()

    # Stage A: Discover filings
    filings = discover_filings()
    print()

    if not args.skip_download:
        # Stage B: Download filings
        download_all_filings(filings)
        print()

    # Stage C: Clean HTML to text
    cleaned = process_all_filings(filings)
    print()

    # Stage D: Write parquet shards
    write_parquet_shards(cleaned)
    print()

    # Stage E: Train tokenizer
    train_tokenizer_10k()
    print()

    print("Done! Ready to train with:")
    print(f"  AUTORESEARCH_CACHE={CACHE_DIR_10K} uv run train.py")
