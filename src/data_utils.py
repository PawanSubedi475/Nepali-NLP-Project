"""
data_utils.py — Dataset Loading, Preprocessing, and Tokenization
=================================================================
Handles:
  - Kaggle dataset download instructions
  - Text preprocessing for Nepali (Devanagari) and English
  - Vocabulary construction and word-index tokenizer
  - AG News loading via HuggingFace datasets
  - Train/val/test splitting with stratification
"""

import re
import os
import glob
import json
import random
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Optional

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Nepali Unicode ranges ────────────────────────────────────────────────────
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]+')
NEPALI_PUNCTUATION = re.compile(r'[।॥,!?;:()\[\]{}"\']+')
LATIN_CHARS = re.compile(r'[a-zA-Z0-9]+')

# ─── Text Preprocessing ───────────────────────────────────────────────────────

def preprocess_nepali(text: str) -> str:
    """
    Preprocess Devanagari Nepali text.

    Steps:
      1. Lowercase (no-op for Devanagari, applied to any Latin chars)
      2. Remove URLs and email addresses
      3. Remove HTML tags
      4. Remove Nepali punctuation (।॥) and special characters
      5. Normalize whitespace

    Note: No stemming/lemmatization applied — XLM-R's SentencePiece handles
    morphological variation implicitly. For CNN/RNN, we keep raw tokens to
    preserve Nepali morphology (agglutinative suffixes carry meaning).
    """
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\.\S+', '', text)          # URLs
    text = re.sub(r'\S+@\S+', '', text)                    # emails
    text = re.sub(r'<[^>]+>', '', text)                    # HTML
    text = NEPALI_PUNCTUATION.sub(' ', text)               # punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_english(text: str) -> str:
    """
    Preprocess English text for AG News.

    Steps:
      1. Lowercase
      2. Remove URLs, HTML
      3. Remove non-alphanumeric characters (keep spaces)
      4. Normalize whitespace
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─── Vocabulary and Word-Index Tokenizer ──────────────────────────────────────

class Vocabulary:
    """
    Build and serialize a word-level vocabulary from training corpus.
    Special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3
    """

    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    SPECIAL = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

    def __init__(self, max_size: int = 50000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._built = False

    def build(self, texts: List[str]) -> 'Vocabulary':
        """Build vocabulary from a list of preprocessed texts."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Start with special tokens
        for i, tok in enumerate(self.SPECIAL):
            self.word2idx[tok] = i
            self.idx2word[i] = tok

        # Add tokens by frequency, up to max_size
        for word, freq in counter.most_common(self.max_size - len(self.SPECIAL)):
            if freq < self.min_freq:
                break
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self._built = True
        print(f"[Vocab] Built: {len(self.word2idx):,} tokens "
              f"(max={self.max_size:,}, min_freq={self.min_freq})")
        return self

    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Convert text to token indices, padding/truncating to max_length."""
        if not self._built:
            raise RuntimeError("Call build() first.")
        tokens = text.split()[:max_length]
        ids = [self.word2idx.get(t, self.UNK) for t in tokens]
        # Pad to max_length
        ids += [self.PAD] * (max_length - len(ids))
        return ids

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx, 'max_size': self.max_size,
                       'min_freq': self.min_freq}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        v = cls(max_size=data['max_size'], min_freq=data['min_freq'])
        v.word2idx = data['word2idx']
        v.idx2word = {int(i): w for w, i in v.word2idx.items()}
        v._built = True
        return v

    def __len__(self):
        return len(self.word2idx)


# ─── Dataset Loaders ─────────────────────────────────────────────────────────

NEPALI_CATEGORY_MAP = {
    'business': 0, 'crime': 1, 'economy': 2, 'education': 3,
    'entertainment': 4, 'health': 5, 'international': 6,
    'politics': 7, 'sports': 8, 'technology': 9
}

# AG News → Nepali category mapping (4 → subset of 10 classes)
AG_TO_NEPALI_MAP = {
    0: 6,  # World      → International
    1: 8,  # Sports     → Sports
    2: 0,  # Business   → Business
    3: 9,  # Sci/Tech   → Technology
}

FOLDER_CATEGORY_MAP = {
    'agriculture': 'health',
    'automobiles': 'technology',
    'bank': 'economy',
    'blog': 'entertainment',
    'business': 'business',
    'economy': 'economy',
    'education': 'education',
    'employment': 'economy',
    'entertainment': 'entertainment',
    'health': 'health',
    'interview': 'politics',
    'literature': 'entertainment',
    'migration': 'international',
    'opinion': 'politics',
    'politics': 'politics',
    'society': 'international',
    'sports': 'sports',
    'technology': 'technology',
    'tourism': 'entertainment',
    'world': 'international',
}


def _is_text_folder_dataset(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    categories = [d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d))]
    if len(categories) < 5:
        return False
    for category in categories:
        if glob.glob(os.path.join(path, category, '*.txt')):
            return True
    return False


def _search_text_dataset_folder(data_dir: str) -> Optional[str]:
    search_root = data_dir if os.path.isdir(data_dir) else os.getcwd()
    candidates = [
        data_dir,
        os.path.join('Dataset', 'nepali_news_dataset_20_categories_large', 'nepali_news_dataset_20_categories_large'),
        os.path.join('Dataset', 'nepali_news_dataset_20_categories_large'),
        os.path.join('Dataset', 'nepali_news_dataset'),
        os.path.join('data', 'nepali_news_dataset_20_categories_large', 'nepali_news_dataset_20_categories_large'),
        os.path.join('data', 'nepali_news_dataset_20_categories_large'),
        os.path.join('data', 'nepali_news_dataset'),
    ]
    for candidate in candidates:
        if candidate and _is_text_folder_dataset(candidate):
            return candidate

    for root, dirs, files in os.walk(search_root):
        if _is_text_folder_dataset(root):
            return root
        if root.count(os.sep) - os.path.abspath(search_root).count(os.sep) > 4:
            dirs[:] = []
    return None


def _prepare_nepali_dataset_from_folders(folder: str) -> Tuple[List, List, List, List, List, List]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError('pandas required: pip install pandas')

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError('scikit-learn required: pip install scikit-learn')

    rows = []
    for category in sorted([d for d in os.listdir(folder)
                            if os.path.isdir(os.path.join(folder, d))]):
        mapped = FOLDER_CATEGORY_MAP.get(category.lower().strip())
        if mapped is None:
            continue
        label = NEPALI_CATEGORY_MAP[mapped]
        for txt_path in glob.glob(os.path.join(folder, category, '*.txt')):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            except Exception:
                continue
            if content:
                rows.append((preprocess_nepali(content), label))

    if not rows:
        raise FileNotFoundError(
            f'No valid text files found in folder dataset: {folder}'
        )

    df = pd.DataFrame(rows, columns=['text', 'label'])
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=SEED
    )
    return (
        train_df['text'].tolist(), train_df['label'].astype(int).tolist(),
        val_df['text'].tolist(), val_df['label'].astype(int).tolist(),
        test_df['text'].tolist(), test_df['label'].astype(int).tolist(),
    )


def load_nepali_dataset(data_dir: str) -> Tuple[List, List, List, List, List, List]:
    """
    Load the Babu (2024) Nepali News Categorization dataset.

    Dataset structure (from Kaggle):
        nepali_news_dataset/
            train.csv   — 48,808 rows (70%)
            val.csv     — 10,459 rows (15%)
            test.csv    — 10,459 rows (15%)
        Columns: ['category', 'title', 'content']

    Returns:
        (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)

    Download instructions:
        kaggle datasets download -d newaribabu/dataset-news-categorization
        unzip dataset-news-categorization.zip -d nepali_news_dataset/
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    splits = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(data_dir, f'{split}.csv')
        if not os.path.exists(path):
            folder_dataset = _search_text_dataset_folder(data_dir)
            if folder_dataset is not None:
                print(f"[Data] No CSV files found in {data_dir}."
                      f" Loading folder dataset from {folder_dataset}.")
                return _prepare_nepali_dataset_from_folders(folder_dataset)
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                f"Download from Kaggle: newaribabu/dataset-news-categorization\n"
                f"Or prepare the folder dataset using: python src/prepare_dataset_from_folders.py\n"
                f"See README.md for full setup instructions."
            )
        df = pd.read_csv(path)
        df['text'] = (df['title'].fillna('') + ' ' + df['content'].fillna('')
                      ).apply(preprocess_nepali)
        df['label'] = df['category'].str.lower().map(NEPALI_CATEGORY_MAP)
        df = df.dropna(subset=['label'])
        splits[split] = (df['text'].tolist(), df['label'].astype(int).tolist())

    return (*splits['train'], *splits['val'], *splits['test'])


def load_ag_news(mapped_to_nepali: bool = True) -> Tuple[List, List]:
    """
    Load AG News dataset via HuggingFace datasets library.
    Maps the 4 AG News categories to their Nepali equivalents if requested.

    Returns: (texts, labels)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets required: pip install datasets")

    print("[Data] Loading AG News from HuggingFace...")
    ds = load_dataset('ag_news', split='train')
    texts = [(preprocess_english(row['text'])) for row in ds]
    labels = [row['label'] for row in ds]

    if mapped_to_nepali:
        labels = [AG_TO_NEPALI_MAP[l] for l in labels]

    print(f"[Data] AG News loaded: {len(texts):,} samples")
    return texts, labels


def create_condition_b_data(
    nepali_texts, nepali_labels,
    english_texts, english_labels,
    nepali_fraction: float = 0.5
) -> Tuple[List, List]:
    """
    Create Condition B training set: English pre-training data + 50% Nepali.

    Strategy:
        1. Use all English data for domain exposure
        2. Randomly sample `nepali_fraction` of Nepali training data
        3. Concatenate and shuffle

    This simulates the cross-lingual transfer scenario where labeled
    Nepali data is scarce but English data is abundant.
    """
    # Sample 50% of Nepali training data (stratified by class)
    n_nepali = int(len(nepali_texts) * nepali_fraction)
    indices = list(range(len(nepali_texts)))
    random.shuffle(indices)
    sampled_idx = indices[:n_nepali]

    combined_texts = (english_texts +
                      [nepali_texts[i] for i in sampled_idx])
    combined_labels = (english_labels +
                       [nepali_labels[i] for i in sampled_idx])

    # Shuffle combined data
    paired = list(zip(combined_texts, combined_labels))
    random.shuffle(paired)
    texts, labels = zip(*paired)

    print(f"[Data] Condition B: {len(english_texts):,} EN + {n_nepali:,} NE "
          f"= {len(texts):,} total samples")
    return list(texts), list(labels)


# ─── Synthetic Data Generator (for simulate mode) ────────────────────────────

def generate_synthetic_samples(
    n_samples: int = 1000,
    n_classes: int = 10,
    vocab_size: int = 5000,
    seq_len: int = 128,
    language: str = 'nepali'
) -> Tuple[List[str], List[int]]:
    """
    Generate synthetic text samples for fast demonstration.
    Each sample is a random sequence of 'words' (integer tokens rendered as strings).
    Used ONLY in --mode simulate; never in actual training.
    """
    texts, labels = [], []
    # Class-specific word distributions to simulate topic coherence
    class_vocab_centers = [
        random.sample(range(vocab_size), 200) for _ in range(n_classes)
    ]

    for _ in range(n_samples):
        label = random.randint(0, n_classes - 1)
        # Draw 70% words from class-specific vocab, 30% random (noise)
        class_words = random.choices(class_vocab_centers[label], k=int(seq_len * 0.7))
        noise_words = random.choices(range(vocab_size), k=seq_len - len(class_words))
        word_ids = class_words + noise_words
        random.shuffle(word_ids)

        if language == 'nepali':
            # Simulate Devanagari tokens
            text = ' '.join(f'शब्द{w}' for w in word_ids)
        else:
            text = ' '.join(f'word{w}' for w in word_ids)

        texts.append(text)
        labels.append(label)

    return texts, labels


def compute_class_weights(labels: List[int], n_classes: int) -> 'np.ndarray':
    """
    Compute inverse-frequency class weights for imbalanced datasets.
    Used with nn.CrossEntropyLoss(weight=...) to handle class imbalance.

    w_c = N / (n_classes × n_c)   where n_c = samples in class c
    """
    counts = Counter(labels)
    N = len(labels)
    weights = np.array([
        N / (n_classes * counts.get(c, 1)) for c in range(n_classes)
    ])
    return weights / weights.sum() * n_classes  # normalize


def get_data_statistics(texts: List[str], labels: List[int],
                        label_names: List[str]) -> Dict:
    """Return descriptive statistics for a dataset split."""
    lengths = [len(t.split()) for t in texts]
    label_dist = Counter(labels)
    return {
        'n_samples': len(texts),
        'n_classes': len(set(labels)),
        'avg_length': round(np.mean(lengths), 1),
        'median_length': round(np.median(lengths), 1),
        'max_length': max(lengths),
        'min_length': min(lengths),
        'class_distribution': {
            label_names[k]: v for k, v in sorted(label_dist.items())
        }
    }
