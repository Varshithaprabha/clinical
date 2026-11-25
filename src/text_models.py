"""BioBERT sentence embeddings (mean-pooled last hidden state)."""
from __future__ import annotations

import numpy as np

from .config import BIOBERT_NAME

_tok = _model = None


def _load():
    global _tok, _model
    if _model is None:
        import torch
        from huggingface_hub import hf_hub_download
        from tokenizers import BertWordPieceTokenizer
        from transformers import PreTrainedTokenizerFast
        # BioBERT ships only vocab.txt; transformers 5 cannot auto-convert it, so build the
        # cased WordPiece tokenizer directly from the vocab file.
        wp = BertWordPieceTokenizer(hf_hub_download(BIOBERT_NAME, "vocab.txt"), lowercase=False, strip_accents=False)
        _tok = PreTrainedTokenizerFast(tokenizer_object=wp._tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                                       cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
        from transformers import BertModel
        _model = BertModel.from_pretrained(BIOBERT_NAME)  # config.json has no model_type -> load as BERT
        _model.eval()
        torch.set_grad_enabled(False)
    return _tok, _model


def embed(texts: list[str], batch_size: int = 32, max_length: int = 128) -> np.ndarray:
    import torch
    tok, model = _load()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy())
    return np.vstack(out).astype(np.float32)
