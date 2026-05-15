"""
models.py — PyTorch Model Architectures
========================================
Implements TextCNN, BiLSTM-Attention, and XLM-R classifier.
All models expose a consistent interface: forward(x) → logits.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Warning] PyTorch not installed — model forward passes unavailable.")


if TORCH_AVAILABLE:

    class TextCNNModel(nn.Module):
        """
        Convolutional Neural Network for text classification (Kim, 2014).

        Architecture:
            Embedding(V, d) → Conv1d(filter_size, n_filters) × K
                            → ReLU → MaxPool
                            → Concat(K × n_filters)
                            → Dropout → Linear(num_classes)

        Multiple filter sizes capture different n-gram features simultaneously.
        Max-over-time pooling (Collobert et al., 2011) extracts the most
        salient feature per filter regardless of sequence position.
        """

        def __init__(self, vocab_size: int, embed_dim: int = 300,
                     num_filters: int = 100, filter_sizes: tuple = (3, 4, 5),
                     num_classes: int = 10, dropout: float = 0.5,
                     pad_idx: int = 0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            # Each Conv1d applies `num_filters` kernels of size (filter_size × embed_dim)
            self.convolutions = nn.ModuleList([
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
                          kernel_size=fs, padding=fs // 2)
                for fs in filter_sizes
            ])
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            # x: (batch, seq_len)
            emb = self.embedding(x)          # (batch, seq_len, embed_dim)
            emb = emb.permute(0, 2, 1)       # (batch, embed_dim, seq_len)

            pooled = []
            for conv in self.convolutions:
                activated = F.relu(conv(emb))    # (batch, num_filters, seq_out)
                # Max-over-time pooling: collapse temporal dimension
                p = F.max_pool1d(activated, activated.size(2)).squeeze(2)
                pooled.append(p)

            cat = torch.cat(pooled, dim=1)   # (batch, K * num_filters)
            out = self.fc(self.dropout(cat)) # (batch, num_classes)
            return out


    class AttentionLayer(nn.Module):
        """
        Additive (Bahdanau-style) self-attention for sequence classification.

        For each hidden state h_t, computes:
            u_t = tanh(W · h_t + b)     [non-linear projection]
            α_t = softmax(v^T · u_t)    [attention score]
            c   = Σ α_t * h_t           [context vector]

        Context vector c captures the most task-relevant parts of the sequence.
        """

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, hidden: 'torch.Tensor',
                    mask: 'torch.Tensor' = None) -> 'torch.Tensor':
            # hidden: (batch, seq_len, hidden_dim)
            u = torch.tanh(self.W(hidden))          # (batch, seq_len, hidden_dim)
            scores = self.v(u).squeeze(-1)           # (batch, seq_len)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            alpha = F.softmax(scores, dim=-1)        # (batch, seq_len)
            context = torch.bmm(alpha.unsqueeze(1), hidden).squeeze(1)
            return context, alpha


    class BiLSTMAttentionModel(nn.Module):
        """
        Bidirectional LSTM with attention (Zhou et al., 2016).

        Architecture:
            Embedding(V, d) → Dropout
            → BiLSTM(hidden_dim, num_layers)       # 2×hidden_dim per token
            → AttentionLayer(2×hidden_dim)          # context vector
            → Dropout → Linear(num_classes)

        BiLSTM captures both past (forward LSTM) and future (backward LSTM)
        context for each token, yielding richer representations than unidirectional.
        """

        def __init__(self, vocab_size: int, embed_dim: int = 300,
                     hidden_dim: int = 256, num_layers: int = 2,
                     num_classes: int = 10, dropout: float = 0.5,
                     pad_idx: int = 0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.dropout = nn.Dropout(dropout)
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.attention = AttentionLayer(hidden_dim * 2)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x: 'torch.Tensor',
                    lengths: 'torch.Tensor' = None) -> 'torch.Tensor':
            emb = self.dropout(self.embedding(x))  # (batch, seq, embed)
            # Pack for efficiency if lengths provided
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    emb, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                output, _ = self.lstm(packed)
                hidden, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            else:
                hidden, _ = self.lstm(emb)          # (batch, seq, 2*hidden)

            context, _ = self.attention(hidden)     # (batch, 2*hidden)
            out = self.fc(self.dropout(context))    # (batch, num_classes)
            return out


    class XLMRClassifier(nn.Module):
        """
        XLM-RoBERTa (base) fine-tuned for sequence classification.

        Architecture:
            XLM-R Encoder (12 layers, 768-dim, 12-head attention)
            → [CLS] token representation
            → Dropout(0.1)
            → Linear(768, num_classes)

        Fine-tuning strategy:
            - Freeze bottom `freeze_layers` transformer layers (default=8)
            - Fine-tune top 4 layers + task head
            - Use lower LR for transformer (1e-5) vs task head (1e-3)

        This selective freezing (Lee et al., 2019) reduces catastrophic
        forgetting while allowing the upper layers to adapt to Nepali text.
        """

        def __init__(self, num_classes: int = 10, dropout: float = 0.1,
                     freeze_layers: int = 8, model_name: str = 'xlm-roberta-base'):
            super().__init__()
            try:
                from transformers import XLMRobertaModel
                self.encoder = XLMRobertaModel.from_pretrained(model_name)
                self._freeze_layers(freeze_layers)
                self.hidden_size = self.encoder.config.hidden_size  # 768
            except ImportError:
                raise ImportError("transformers library required: pip install transformers")

            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.hidden_size, num_classes)

        def _freeze_layers(self, n: int):
            """Freeze the first n transformer encoder layers."""
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < n:
                    for param in layer.parameters():
                        param.requires_grad = False

        def forward(self, input_ids: 'torch.Tensor',
                    attention_mask: 'torch.Tensor') -> 'torch.Tensor':
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            out = self.classifier(self.dropout(cls_repr))  # (batch, num_classes)
            return out

        def get_optimizer_groups(self, lr_encoder: float = 1e-5, lr_head: float = 1e-3):
            """Return parameter groups with different LRs for encoder vs head."""
            return [
                {'params': self.encoder.parameters(), 'lr': lr_encoder},
                {'params': self.classifier.parameters(), 'lr': lr_head},
            ]


    class NepaliNewsDataset(Dataset):
        """
        PyTorch Dataset for Nepali/English news classification.
        Handles tokenization and encoding for CNN/RNN/Transformer models.
        """

        def __init__(self, texts, labels, tokenizer=None, max_len: int = 512,
                     model_type: str = 'cnn'):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.model_type = model_type  # 'cnn', 'rnn', 'transformer'

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]

            if self.model_type == 'transformer' and self.tokenizer:
                # HuggingFace tokenizer path
                enc = self.tokenizer(
                    text, max_length=self.max_len, padding='max_length',
                    truncation=True, return_tensors='pt'
                )
                return {
                    'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                    'label': torch.tensor(label, dtype=torch.long)
                }
            else:
                # Word-index path for CNN/RNN
                ids = self.tokenizer.encode(text, max_length=self.max_len)
                ids = torch.tensor(ids, dtype=torch.long)
                return {'input_ids': ids, 'label': torch.tensor(label, dtype=torch.long)}


    def count_parameters(model: nn.Module) -> int:
        """Count trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def build_model(model_name: str, vocab_size: int = 50000,
                    num_classes: int = 10, **kwargs) -> nn.Module:
        """Factory function for model construction."""
        if model_name == 'TextCNN':
            return TextCNNModel(vocab_size=vocab_size, num_classes=num_classes, **kwargs)
        elif model_name == 'BiLSTM-Attn':
            return BiLSTMAttentionModel(vocab_size=vocab_size, num_classes=num_classes, **kwargs)
        elif model_name == 'XLM-R':
            return XLMRClassifier(num_classes=num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")

# ─── Make functions available even if torch wasn't imported initially ────────
else:
    # Fallback stub when torch is not available
    def build_model(model_name: str, vocab_size: int = 50000,
                    num_classes: int = 10, **kwargs):
        raise RuntimeError(
            "PyTorch not installed. Please install: pip install torch transformers"
        )
    
    class NepaliNewsDataset:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch not installed. Please install: pip install torch transformers"
            )
