in this repo i post my daily learnings
# Tokenization Strategies for BERT-Based Text Classification
BERT-Based model trained on news classification dataset
## Results Summary

| Method | Training Time |
|---|---|
| Normal (Padded) Tokenization | 150s |
| Bucket Batch Tokenization | 110s |
| Packed Sequence Tokenization | 112s |

---

## 1. Normal (Padded) Tokenization — 150s

### What it is
Every text in a batch is tokenized and padded to the length of the **longest
sequence in that batch**. If one article has 480 tokens and the rest have 100,
all sequences in that batch are padded to 480.

### How it works
```
Batch:
[a1 a2 a3 PAD PAD PAD PAD PAD PAD PAD]   ← padded to 480
[b1 b2 b3 b4  b5  PAD PAD PAD PAD PAD]   ← padded to 480
[c1 c2 c3 c4  c5  c6  c7  PAD PAD PAD]   ← padded to 480
```

### Why it is the slowest (150s)
- BERT computes attention over **every token including PAD tokens**, wasting
  compute on tokens that carry no information.
- With a max length of 512 and an average text length of ~183 tokens, roughly
  **64% of each sequence is padding** in the worst case.
- All that wasted attention computation adds up across 112 batches per epoch.

---

## 2. Bucket Batch Tokenization — 110s ✅ Fastest

### What it is
Texts are grouped into **length buckets** before tokenization. Each bucket only
pads sequences to the maximum length of that bucket, not the global maximum.

### How it works
```
Bucket 1 (0–128 tokens)   → pad to 128
Bucket 2 (129–256 tokens) → pad to 256
Bucket 3 (257–512 tokens) → pad to 512

Bucket 1 batch:
[a1 a2 a3 PAD PAD]   ← padded to 128, not 512
[b1 b2 b3 b4  PAD]   ← padded to 128, not 512
```

### Why it is the fastest (110s)
- Given the dataset's token length distribution (25th percentile = 136,
  median = 183, 75th percentile = 260), the **majority of texts fall in
  Bucket 1 and Bucket 2**.
- Short texts are never padded to 512 — they only pad to 128 or 256.
- This roughly **halves the attention computation** for most batches compared
  to normal tokenization.
- Tokenization itself is still done in batches (fast), with no extra overhead.

---

## 3. Packed Sequence Tokenization — 112s

### What it is
Multiple short texts are packed into a **single 512-token sequence** back to
back, separated by `[SEP]` boundaries. Near-zero padding waste.

### How it works
```
Normal padding (wasteful):
[a1 a2 a3 PAD PAD PAD PAD PAD]
[b1 b2 b3 b4  b5  PAD PAD PAD]

Packed sequences (efficient):
[CLS a1 a2 a3 SEP CLS b1 b2 b3 b4 b5 SEP CLS c1 c2 ...]
```
`token_type_ids` alternate between 0 and 1 across segments so BERT can
distinguish one article from the next.

### Why it is slightly slower than bucket batching (112s vs 110s)
- The **compression ratio achieved was only 1.68x** (1,780 samples → 1,062
  packed sequences). The break-even point for packing is typically ~2.5x.
- Tokenization happens **per sample** (1,780 individual calls) rather than in
  batches, adding Python-level overhead before training even starts.
- A **Python loop** builds the packs one token at a time — slow compared to
  the batched tensor operations used in bucket batching.
- With an average token length of ~183, only ~2 articles fit per 512-token
  sequence on average, which is not enough to overcome the packing overhead.

### When packed sequence training wins
Packing outperforms bucket batching when:
- Average token length is **> 300 tokens** (more padding to eliminate)
- Compression ratio is **> 2.5x**
- The packing step is **precomputed offline** and saved to disk, removing all
  overhead from the training loop

---

## Key Takeaway

> Both bucket batching and packed sequence training **only reduce training
> time — not accuracy**. The model learns the same information either way,
> just more or less efficiently.

For this dataset (avg ~183 tokens, 1,780 training samples),
**bucket batch tokenization is the optimal choice** — simplest to implement,
fastest to train, and equally accurate.
