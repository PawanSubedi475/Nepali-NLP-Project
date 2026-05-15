"""
prepare_dataset_from_folders.py — Prepare Nepali News Dataset (From Folder Structure)
======================================================================================
Use this when you have the dataset as folders with text files (not CSV).
It will:
  1. Read all .txt files from category folders
  2. Map 20 categories to our 10-class schema
  3. Split into train / val / test (70 / 15 / 15)
  4. Save as train.csv, val.csv, test.csv in nepali_news_dataset/

Usage:
    python src/prepare_dataset_from_folders.py
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

SEED = 42
np.random.seed(SEED)

# ── Target categories ────────────────────────────────────────────────────────
TARGET_CATEGORIES = [
    'business','crime','economy','education','entertainment',
    'health','international','politics','sports','technology'
]

# ── Category mapping — covers 20-category Nepali News Dataset ────────────────
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

# ── Step 1: Find category folders ────────────────────────────────────────────
def find_dataset_folder():
    search_paths = [
        'Dataset/nepali_news_dataset_20_categories_large/nepali_news_dataset_20_categories_large',
        'Dataset/nepali_news_dataset_20_categories_large',
        'Dataset',
    ]
    
    for path in search_paths:
        if os.path.isdir(path):
            # Check if it has category folders
            contents = os.listdir(path)
            if any(f in contents for f in ['Agriculture', 'Automobiles', 'Bank', 'Business']):
                return path
    
    return None

# ── Step 2: Read all text files ──────────────────────────────────────────────
def read_text_files(dataset_folder):
    data = []
    category_counts = Counter()
    
    # Get all category folders
    category_folders = [d for d in os.listdir(dataset_folder) 
                       if os.path.isdir(os.path.join(dataset_folder, d))]
    
    print(f"\n[Step 1] Found {len(category_folders)} category folders:")
    for cat_folder in sorted(category_folders):
        folder_path = os.path.join(dataset_folder, cat_folder)
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        print(f"    {cat_folder:<20} -> {len(txt_files):,} files")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:  # Only add non-empty files
                    data.append({
                        'category': cat_folder,
                        'text': content,
                    })
                    category_counts[cat_folder] += 1
            except Exception as e:
                print(f"      [Warning] Failed to read {txt_file}: {e}")
    
    return data, category_counts

# ── Step 3: Map categories ──────────────────────────────────────────────────
def map_categories_df(df):
    df['category_lower'] = df['category'].str.lower().str.strip()
    df['category_mapped'] = df['category_lower'].map(CATEGORY_ALIASES)
    
    n_mapped = df['category_mapped'].notna().sum()
    n_total = len(df)
    
    print(f"\n[Step 2] Category mapping: {n_mapped}/{n_total} ({100*n_mapped/n_total:.1f}%)")
    print("\n  Folder categories found:")
    
    for cat, count in sorted(Counter(df['category']).items()):
        mapped_to = CATEGORY_ALIASES.get(cat.lower().strip(), '-- UNMAPPED --')
        print(f"    {cat:<20} -> {mapped_to:<20} ({count:,} files)")
    
    return df

# ── Step 4: Stratified split ────────────────────────────────────────────────
def stratified_split(df, train_frac=0.70, val_frac=0.15):
    from sklearn.model_selection import train_test_split
    
    train_df, temp = train_test_split(df, test_size=1-train_frac,
                                      stratify=df['category_mapped'], random_state=SEED)
    rel_val = val_frac / (1 - train_frac)
    val_df, test_df = train_test_split(temp, test_size=1-rel_val,
                                       stratify=temp['category_mapped'], random_state=SEED)
    return train_df, val_df, test_df

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Nepali News Dataset — Prepare from Folders")
    print("=" * 70)
    
    # Step 1: Find dataset folder
    dataset_folder = find_dataset_folder()
    if dataset_folder is None:
        print("\n[ERROR] Cannot find dataset folder!")
        print("\nExpected structure:")
        print("  Dataset/nepali_news_dataset_20_categories_large/nepali_news_dataset_20_categories_large/")
        print("    Agriculture/")
        print("    Automobiles/")
        print("    ... (other categories)")
        sys.exit(1)
    
    print(f"\n[Found] Dataset folder: {dataset_folder}")
    
    # Step 2: Read all text files
    print("\n[Loading] Reading all text files...")
    data, category_counts = read_text_files(dataset_folder)
    
    if not data:
        print("[ERROR] No text files found in dataset!")
        sys.exit(1)
    
    print(f"\n[Step 1 Summary] Total files read: {len(data):,}")
    
    # Step 3: Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"[Step 1 Summary] DataFrame shape: {len(df):,} rows")
    
    # Step 4: Map categories
    df = map_categories_df(df)
    
    # Drop unmapped
    df = df.dropna(subset=['category_mapped'])
    df['category'] = df['category_mapped'].str.lower().str.strip()
    
    print(f"\n[Step 2 Summary] Valid rows after mapping: {len(df):,}")
    print("  Final distribution:")
    for cat, cnt in sorted(Counter(df['category']).items(), key=lambda x: -x[1]):
        bar = '|' * (cnt // 200)
        print(f"    {cat:<20} {cnt:>6,}  {bar}")
    
    # Prepare final columns
    df['title'] = df['text'].str[:120]
    df['content'] = df['text']
    
    # Step 5: Split
    print(f"\n[Step 3] Splitting 70% / 15% / 15%...")
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("[ERROR] Run: pip install scikit-learn")
        sys.exit(1)
    
    train_df, val_df, test_df = stratified_split(df)
    
    # Step 6: Save
    output_dir = 'nepali_news_dataset'
    os.makedirs(output_dir, exist_ok=True)
    
    for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
        path = os.path.join(output_dir, f'{name}.csv')
        split[['category', 'title', 'content']].to_csv(path, index=False, encoding='utf-8')
        print(f"  [Saved] {name}.csv -> {len(split):,} rows  [{path}]")

    print(f"\n{'='*70}")
    print(f"  Dataset prepared successfully!")
    print(f"  train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")
    print(f"{'='*70}")
    print(f"\n  Next -> python src/main.py --mode full --data-dir nepali_news_dataset")

if __name__ == '__main__':
    main()
