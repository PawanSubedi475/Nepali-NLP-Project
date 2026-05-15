"""
prepare_dataset.py — Auto-detect & Prepare Nepali News Dataset (Large)
=======================================================================
Run this ONCE after downloading the dataset from Kaggle.
It will:
  1. Detect your CSV file automatically (works for the 20-category large dataset)
  2. Map all categories to our 10-class schema
  3. Split into train / val / test (70 / 15 / 15)
  4. Save as train.csv, val.csv, test.csv in nepali_news_dataset/

Usage:
    python src/prepare_dataset.py
"""

import os, sys, glob
import pandas as pd
import numpy as np
from collections import Counter

SEED = 42
np.random.seed(SEED)

# ── Where to look for downloaded files ──────────────────────────────────────
SEARCH_DIRS = ['Dataset', 'dataset', 'nepali_news_dataset', 'data', '.']
OUTPUT_DIR  = 'nepali_news_dataset'

# ── Target categories ────────────────────────────────────────────────────────
TARGET_CATEGORIES = [
    'business','crime','economy','education','entertainment',
    'health','international','politics','sports','technology'
]

# ── Category mapping — covers 20-category Nepali News Dataset Large ──────────
CATEGORY_ALIASES = {
    # Business
    'business':'business', 'व्यापार':'business', 'व्यवसाय':'business',
    'biznez':'business', 'biznes':'business', 'commerce':'business',

    # Economy / Finance
    'economy':'economy', 'economic':'economy', 'अर्थ':'economy',
    'अर्थतन्त्र':'economy', 'artha':'economy', 'finance':'economy',
    'financial':'economy', 'banking':'economy', 'bank':'economy',
    'budget':'economy', 'investment':'economy', 'market':'economy',
    'infrastructure':'economy', 'development':'economy', 'विकास':'economy',
    'labor':'economy', 'trade':'economy',

    # Politics / Government
    'politics':'politics', 'political':'politics', 'राजनीति':'politics',
    'राजनैतिक':'politics', 'rajniti':'politics', 'national':'politics',
    'government':'politics', 'election':'politics', 'निर्वाचन':'politics',
    'party':'politics', 'opinion':'politics', 'editorial':'politics',
    'interview':'politics', 'diplomacy':'politics', 'policy':'politics',

    # Sports
    'sports':'sports', 'sport':'sports', 'खेलकुद':'sports',
    'khelkud':'sports', 'football':'sports', 'cricket':'sports',
    'game':'sports', 'athletics':'sports',

    # Technology / Science
    'technology':'technology', 'tech':'technology', 'science':'technology',
    'science/tech':'technology', 'sci/tech':'technology', 'विज्ञान':'technology',
    'प्रविधि':'technology', 'prawidhi':'technology', 'it':'technology',
    'digital':'technology', 'innovation':'technology', 'space':'technology',

    # International / World
    'international':'international', 'world':'international', 'global':'international',
    'विश्व':'international', 'अन्तर्राष्ट्रिय':'international',
    'antarrashtriya':'international', 'foreign':'international',
    'abroad':'international', 'social':'international', 'society':'international',
    'community':'international', 'religion':'international', 'धर्म':'international',
    'migration':'international', 'human rights':'international',

    # Entertainment / Culture
    'entertainment':'entertainment', 'culture':'entertainment', 'art':'entertainment',
    'music':'entertainment', 'film':'entertainment', 'movie':'entertainment',
    'celebrity':'entertainment', 'मनोरञ्जन':'entertainment', 'manoranjan':'entertainment',
    'lifestyle':'entertainment', 'fashion':'entertainment', 'travel':'entertainment',
    'food':'entertainment', 'tourism':'entertainment', 'पर्यटन':'entertainment',

    # Health / Environment / Agriculture
    'health':'health', 'medical':'health', 'स्वास्थ्य':'health',
    'swasthya':'health', 'corona':'health', 'covid':'health',
    'environment':'health', 'nature':'health', 'agriculture':'health',
    'farming':'health', 'कृषि':'health', 'climate':'health', 'weather':'health',

    # Education
    'education':'education', 'शिक्षा':'education', 'siksha':'education',
    'school':'education', 'university':'education', 'exam':'education',
    'scholarship':'education', 'gender':'education', 'women':'education',
    'child':'education', 'youth':'education',

    # Crime / Law / Disaster
    'crime':'crime', 'criminal':'crime', 'अपराध':'crime', 'apradh':'crime',
    'law':'crime', 'court':'crime', 'police':'crime', 'corruption':'crime',
    'accident':'crime', 'disaster':'crime', 'earthquake':'crime', 'flood':'crime',
}

# ── Step 1: Find CSV files ────────────────────────────────────────────────────
def find_csv_files():
    found = []
    for d in SEARCH_DIRS:
        found += glob.glob(os.path.join(d, '**', '*.csv'), recursive=True)
        found += glob.glob(os.path.join(d, '*.csv'))
        found += glob.glob(os.path.join(d, '**', '*.tsv'), recursive=True)

    # Exclude known non-dataset files
    EXCLUDE = ['results_summary.csv', 'results.csv', 'all_results']
    found = list(set(
        f for f in found
        if not any(ex in f for ex in EXCLUDE)
        and ('results' + os.sep) not in f
    ))
    # Sort by size — largest file is the actual dataset
    found = sorted(found, key=lambda f: os.path.getsize(f), reverse=True)

    print(f"\n[Step 1] Found {len(found)} data file(s):")
    for f in found:
        print(f"         {f}  ({os.path.getsize(f)/1024:.0f} KB)")
    return found

# ── Step 2: Detect columns ───────────────────────────────────────────────────
def detect_columns(df):
    cols_lower = [c.lower().strip() for c in df.columns]
    col_map = {c.lower().strip(): orig for c, orig in zip(cols_lower, df.columns)}

    cat_candidates = ['category','label','class','topic','type',
                      'categories','news_type','tag','section',
                      'news_category','cat','genre','subject','labels']
    cat_col = next((col_map[c] for c in cat_candidates if c in col_map), None)

    # Fallback: column with fewest unique string values (2–30)
    if cat_col is None:
        for orig in df.columns:
            n = df[orig].nunique()
            if df[orig].dtype == object and 2 <= n <= 30:
                cat_col = orig
                break

    text_candidates = ['content','body','text','article','news',
                       'description','abstract','paragraph','detail']
    title_candidates = ['title','headline','heading','subject','name']

    text_col  = next((col_map[c] for c in text_candidates if c in col_map), None)
    title_col = next((col_map[c] for c in title_candidates if c in col_map), None)

    # Fallback: longest avg-length string column (not the category)
    if text_col is None:
        str_cols = [c for c in df.columns if df[c].dtype == object and c != cat_col]
        if str_cols:
            text_col = max(str_cols, key=lambda c: df[c].fillna('').str.len().mean())

    return cat_col, title_col, text_col

# ── Step 3: Map categories ────────────────────────────────────────────────────
def map_categories(series):
    mapped = series.str.lower().str.strip().map(CATEGORY_ALIASES)
    n_mapped, n_total = mapped.notna().sum(), len(mapped)
    print(f"\n[Step 3] Category mapping: {n_mapped}/{n_total} "
          f"({100*n_mapped/n_total:.1f}%)")
    print("\n  Raw categories found:")
    for cat, count in sorted(Counter(series.str.lower().str.strip()).items(),
                             key=lambda x: -x[1]):
        mapped_to = CATEGORY_ALIASES.get(cat, '-- UNMAPPED (copy & send to Claude) --')
        print(f"    {cat:<35} -> {mapped_to:<20} ({count:,} rows)")
    return mapped

# ── Step 4: Stratified split ─────────────────────────────────────────────────
def stratified_split(df, train_frac=0.70, val_frac=0.15):
    from sklearn.model_selection import train_test_split
    train_df, temp = train_test_split(df, test_size=1-train_frac,
                                      stratify=df['category'], random_state=SEED)
    rel_val = val_frac / (1 - train_frac)
    val_df, test_df = train_test_split(temp, test_size=1-rel_val,
                                       stratify=temp['category'], random_state=SEED)
    return train_df, val_df, test_df

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Nepali News Dataset — Auto-Prepare Script")
    print("=" * 60)

    # Step 1
    csv_files = find_csv_files()
    if not csv_files:
        print("\n[ERROR] No CSV files found!")
        print("Make sure you ran:")
        print("  kaggle datasets download -d ashokpant/nepali-news-dataset-large")
        print("  Expand-Archive / unzip the downloaded zip first")
        sys.exit(1)

    # Step 2 — load largest file
    main_csv = csv_files[0]
    print(f"\n[Step 2] Loading: {main_csv}")

    # Try comma first, then tab
    try:
        df = pd.read_csv(main_csv, encoding='utf-8', on_bad_lines='skip')
        if df.shape[1] == 1:           # probably tab-separated
            df = pd.read_csv(main_csv, sep='\t', encoding='utf-8', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(main_csv, encoding='latin-1', on_bad_lines='skip')

    print(f"         Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"         Columns: {list(df.columns)}")
    print(f"\n  First 3 rows preview:")
    print(df.head(3).to_string())

    # Step 3 — detect columns
    cat_col, title_col, text_col = detect_columns(df)
    print(f"\n[Step 3] Detected:")
    print(f"         Category -> '{cat_col}'")
    print(f"         Title    -> '{title_col}'")
    print(f"         Content  -> '{text_col}'")

    if cat_col is None:
        print("\n[ERROR] Cannot detect category column.")
        print("Send the Columns list printed above to Claude for a fix.")
        sys.exit(1)

    if text_col is None and title_col is None:
        print("\n[ERROR] Cannot detect any text column.")
        print("Send the Columns list printed above to Claude for a fix.")
        sys.exit(1)

    # Build combined text
    if title_col and text_col:
        df['text'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
    elif text_col:
        df['text'] = df[text_col].fillna('')
    else:
        df['text'] = df[title_col].fillna('')

    # Map categories
    df['category'] = map_categories(df[cat_col])
    df = df.dropna(subset=['category'])
    df['category'] = df['category'].str.lower().str.strip()

    print(f"\n[Step 4] Valid rows after mapping: {len(df):,}")
    print("  Final distribution:")
    for cat, cnt in sorted(Counter(df['category']).items(), key=lambda x: -x[1]):
        bar = '|' * (cnt // 200)
        print(f"    {cat:<20} {cnt:>6,}  {bar}")

    df['title']   = df['text'].str[:120]
    df['content'] = df['text']

    # Split
    print(f"\n[Step 5] Splitting 70% / 15% / 15% ...")
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("[ERROR] Run: pip install scikit-learn")
        sys.exit(1)

    train_df, val_df, test_df = stratified_split(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, split in [('train',train_df),('val',val_df),('test',test_df)]:
        path = os.path.join(OUTPUT_DIR, f'{name}.csv')
        split[['category','title','content']].to_csv(path, index=False, encoding='utf-8')
        print(f"  Saved {name}.csv -> {len(split):,} rows  [{path}]")

    print(f"\n{'='*60}")
    print(f"  Dataset prepared successfully!")
    print(f"  train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")
    print(f"{'='*60}")
    print(f"\n  Next -> python src/main.py --mode full --data-dir nepali_news_dataset")

if __name__ == '__main__':
    main()
