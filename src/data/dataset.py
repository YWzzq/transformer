"""
Dataset and data processing for IWSLT2017 EN-DE translation.
"""
import re
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from collections import Counter
import pickle
import os


class Vocabulary:
    """
    Vocabulary for mapping tokens to indices and vice versa.
    
    Special tokens:
    - <pad>: Padding token (index 0)
    - <bos>: Begin-of-sequence token (index 1)
    - <eos>: End-of-sequence token (index 2)
    - <unk>: Unknown token (index 3)
    """
    
    PAD_TOKEN = '<pad>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    
    def __init__(self, max_vocab_size: int = 32000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        # Special tokens
        self.token2idx = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
    
    def build_vocabulary(self, sentences: List[List[str]]):
        """Build vocabulary from tokenized sentences."""
        # Count token frequencies
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence)
        
        # Sort by frequency and add to vocabulary
        for token, freq in counter.most_common(self.max_vocab_size - 4):
            if freq >= self.min_freq:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens."""
        return [self.idx2token.get(idx, self.UNK_TOKEN) for idx in indices]
    
    def __len__(self):
        return len(self.token2idx)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'max_vocab_size': self.max_vocab_size,
                'min_freq': self.min_freq
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(data['max_vocab_size'], data['min_freq'])
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        return vocab


def tokenize(text: str, lang: str = 'en') -> List[str]:
    """
    Simple tokenization: lowercase and split by whitespace.
    
    Args:
        text: Input text
        lang: Language ('en' or 'de')
    
    Returns:
        List of tokens
    """
    # Remove XML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    
    # Lowercase and split
    text = text.lower().strip()
    
    # Simple whitespace tokenization
    tokens = text.split()
    
    return tokens


def load_iwslt_data(
    data_dir: str,
    max_samples: int = None,
    max_len: int = 128
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Load IWSLT2017 EN-DE training and validation data.
    
    Args:
        data_dir: Path to data directory
        max_samples: Maximum number of samples to load (for quick testing)
        max_len: Maximum sequence length (filter longer sequences)
    
    Returns:
        train_pairs: List of (src, tgt) sentence pairs for training
        val_pairs: List of (src, tgt) sentence pairs for validation
    """
    # Load training data
    train_en_path = os.path.join(data_dir, 'train.tags.en-de.en')
    train_de_path = os.path.join(data_dir, 'train.tags.en-de.de')
    
    print(f"Loading training data from {data_dir}...")
    
    train_en_lines = []
    train_de_lines = []
    
    # Read English file
    with open(train_en_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip XML tags and metadata lines
            if not line.startswith('<'):
                train_en_lines.append(line.strip())
    
    # Read German file
    with open(train_de_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('<'):
                train_de_lines.append(line.strip())
    
    # Create pairs and filter
    train_pairs = []
    for en, de in zip(train_en_lines, train_de_lines):
        if en and de:  # Skip empty lines
            en_tokens = tokenize(en, 'en')
            de_tokens = tokenize(de, 'de')
            
            # Filter by length
            if len(en_tokens) <= max_len and len(de_tokens) <= max_len:
                train_pairs.append((en, de))
        
        if max_samples and len(train_pairs) >= max_samples:
            break
    
    # Use a portion of training data for validation (last 2000 samples)
    val_size = min(2000, len(train_pairs) // 10)
    val_pairs = train_pairs[-val_size:]
    train_pairs = train_pairs[:-val_size]
    
    print(f"Loaded {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")
    
    return train_pairs, val_pairs


class TranslationDataset(Dataset):
    """
    Dataset for machine translation.
    
    Args:
        pairs: List of (source, target) sentence pairs
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        max_len: Maximum sequence length
    """
    
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: int = 128
    ):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        
        # Tokenize
        src_tokens = tokenize(src_text, 'en')[:self.max_len]
        tgt_tokens = tokenize(tgt_text, 'de')[:self.max_len]
        
        # Convert to indices and add BOS/EOS
        src_indices = [self.src_vocab.bos_idx] + \
                      self.src_vocab.encode(src_tokens) + \
                      [self.src_vocab.eos_idx]
        
        tgt_indices = [self.tgt_vocab.bos_idx] + \
                      self.tgt_vocab.encode(tgt_tokens) + \
                      [self.tgt_vocab.eos_idx]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def collate_fn(batch, pad_idx=0):
    """
    Collate function for DataLoader with padding.
    
    Args:
        batch: List of samples from dataset
        pad_idx: Padding index
    
    Returns:
        Dictionary with batched and padded tensors
    """
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    # Pad sequences
    src_lengths = torch.tensor([len(s) for s in src_batch])
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch])
    
    # Pad to max length in batch
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_idx
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=pad_idx
    )
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths
    }


def build_vocabularies(
    train_pairs: List[Tuple[str, str]],
    max_vocab_size: int = 32000,
    min_freq: int = 2,
    save_dir: str = None
) -> Tuple[Vocabulary, Vocabulary]:
    """
    Build source and target vocabularies from training data.
    
    Args:
        train_pairs: List of (source, target) sentence pairs
        max_vocab_size: Maximum vocabulary size
        min_freq: Minimum token frequency
        save_dir: Directory to save vocabularies
    
    Returns:
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
    """
    print("Building vocabularies...")
    
    # Tokenize all sentences
    src_sentences = [tokenize(src, 'en') for src, _ in train_pairs]
    tgt_sentences = [tokenize(tgt, 'de') for _, tgt in train_pairs]
    
    # Build vocabularies
    src_vocab = Vocabulary(max_vocab_size, min_freq)
    tgt_vocab = Vocabulary(max_vocab_size, min_freq)
    
    src_vocab.build_vocabulary(src_sentences)
    tgt_vocab.build_vocabulary(tgt_sentences)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Save vocabularies
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        src_vocab.save(os.path.join(save_dir, 'src_vocab.pkl'))
        tgt_vocab.save(os.path.join(save_dir, 'tgt_vocab.pkl'))
        print(f"Vocabularies saved to {save_dir}")
    
    return src_vocab, tgt_vocab


def create_dataloaders(
    train_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    batch_size: int = 32,
    max_len: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_pairs: Training sentence pairs
        val_pairs: Validation sentence pairs
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        batch_size: Batch size
        max_len: Maximum sequence length
        num_workers: Number of data loading workers
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    # Create datasets
    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab, max_len)
    val_dataset = TranslationDataset(val_pairs, src_vocab, tgt_vocab, max_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx),
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def load_multi30k_data(
    data_dir: str,
    max_samples: int = None,
    max_len: int = 128
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Load Multi30K EN-DE training and validation data.
    
    Args:
        data_dir: Path to data directory (e.g., 'data/datasets')
        max_samples: Maximum number of samples to load (for quick testing)
        max_len: Maximum sequence length (filter longer sequences)
    
    Returns:
        train_pairs: List of (src, tgt) sentence pairs for training
        val_pairs: List of (src, tgt) sentence pairs for validation
    """
    # Load training data
    train_en_path = os.path.join(data_dir, 'train', 'train.en')
    train_de_path = os.path.join(data_dir, 'train', 'train.de')
    
    print(f"Loading Multi30K training data from {data_dir}...")
    
    with open(train_en_path, 'r', encoding='utf-8') as f:
        train_en_lines = [line.strip() for line in f if line.strip()]
    
    with open(train_de_path, 'r', encoding='utf-8') as f:
        train_de_lines = [line.strip() for line in f if line.strip()]
    
    # Create pairs and filter
    train_pairs = []
    for en, de in zip(train_en_lines, train_de_lines):
        if not en or not de:
            continue
        
        # Tokenize and filter by length
        en_tokens = tokenize(en, 'en')
        de_tokens = tokenize(de, 'de')
        
        if len(en_tokens) <= max_len and len(de_tokens) <= max_len:
            train_pairs.append((en, de))
        
        if max_samples and len(train_pairs) >= max_samples:
            break
    
    # Load validation data
    val_en_path = os.path.join(data_dir, 'valid', 'val.en')
    val_de_path = os.path.join(data_dir, 'valid', 'val.de')
    
    print(f"Loading Multi30K validation data...")
    
    with open(val_en_path, 'r', encoding='utf-8') as f:
        val_en_lines = [line.strip() for line in f if line.strip()]
    
    with open(val_de_path, 'r', encoding='utf-8') as f:
        val_de_lines = [line.strip() for line in f if line.strip()]
    
    val_pairs = []
    for en, de in zip(val_en_lines, val_de_lines):
        if not en or not de:
            continue
        
        en_tokens = tokenize(en, 'en')
        de_tokens = tokenize(de, 'de')
        
        if len(en_tokens) <= max_len and len(de_tokens) <= max_len:
            val_pairs.append((en, de))
    
    print(f"Loaded {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")
    
    return train_pairs, val_pairs

