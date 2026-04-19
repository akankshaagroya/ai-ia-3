# Handwritten Equation Recognition System

End-to-end deep learning pipeline that recognizes handwritten mathematical equations from images and computes their answers.

**Input:** Photo of handwritten equation → **Output:** LaTeX equation + computed result

## Architecture

**Seq2Seq with Attention (CNN Encoder + LSTM Decoder)**

```
Handwritten Image (640×480)
    ↓
CNN Encoder (3 conv layers: 16→32→64 channels)
    ↓
128-dim Feature Vector
    ↓
Attention Mechanism
    ↓
LSTM Decoder (embedding + attention-aware decoding)
    ↓
LaTeX Tokens (greedy generation)
    ↓
Mathematical Expression
    ↓
Safe Evaluation → Numeric Answer
```

**Why seq2seq?**
- Avoids broken contour-based symbol segmentation that fails on cramped/connected handwriting
- Learns features directly from images without needing perfect symbol separation
- Generates entire equations token-by-token with attention over image regions

## Project Structure

```
equation_solver/
├── __init__.py                 Package marker
├── mathwriting_loader.py       INKML parsing, vocabulary, LaTeX tokenization
└── seq2seq_model.py            CNN encoder + LSTM decoder + attention

train_mathwriting.py            Training entry point
test.py                         Inference on equation images
requirements.txt                Dependencies
README.md                       This file
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- PyTorch 2.0+
- torchvision
- Pillow
- NumPy

## Usage

### Training

```bash
python3 train_mathwriting.py \
  --train-samples 10000 \
  --val-samples 1000 \
  --epochs 5 \
  --batch-size 16 \
  --device cuda
```

**Arguments:**
- `--train-samples`: Number of training samples (default: 50000)
- `--val-samples`: Number of validation samples (default: 5000)
- `--epochs`: Training epochs (default: 5)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--model-path`: Where to save trained model (default: `models/seq2seq_mathwriting.pt`)
- `--device`: `cuda` or `cpu` (default: auto-detect)

**Output:** Trained model saved to `models/seq2seq_mathwriting.pt` (~151MB)

### Inference

**Test on directory of images:**
```bash
python3 test.py --dir equations
```

**Test single image:**
```bash
python3 test.py --image path/to/equation.png
```

**Output example:**
```
→ equation.png
  Equation:  2 + 3 \times 4
  Expression: 2+3*4
  Answer:    14
```

**Supported formats:** `.png`, `.jpg`, `.jpeg`, `.inkml`

## Key Files

### `equation_solver/mathwriting_loader.py`
- `parse_inkml_strokes(inkml_path)` — Extract strokes and LaTeX from INKML files
- `strokes_to_image(strokes)` — Render strokes to 640×480 grayscale image
- `build_vocabulary(samples)` — Create token→index mapping from LaTeX expressions
- `tokenize_latex(latex_str)` — Break LaTeX into meaningful tokens
- `encode_sequence(tokens, vocab, max_len)` — Convert tokens to indices
- `decode_sequence(indices, idx2token)` — Convert indices back to tokens
- `load_mathwriting_samples(split, max_samples)` — Load dataset samples

### `equation_solver/seq2seq_model.py`
- `CNNEncoder` — 3-layer CNN that extracts features from equation images
- `Attention` — Attention mechanism over encoder outputs
- `LSTMDecoder` — LSTM with attention-aware decoding
- `Seq2Seq` — Complete model combining encoder and decoder
- `build_seq2seq_model(vocab_size, device)` — Create and initialize model
- `save_model(model, vocab, idx2token, path)` — Save checkpoint
- `load_model(path, device)` — Load trained model with vocabulary

### `train_mathwriting.py`
Training script with:
- `MathWritingDataset` — PyTorch Dataset wrapper for INKML samples
- `train_epoch()` — Single training epoch with teacher forcing
- `eval_epoch()` — Validation with loss tracking
- Early stopping, learning rate scheduling, checkpointing

### `test.py`
Inference script that:
- Loads trained model
- Processes images (resize to 640×480, normalize)
- Generates LaTeX using greedy beam search
- Parses LaTeX to evaluable expression
- Safely evaluates math (restricted builtins, no code execution)
- Prints equation and answer

## How It Works

1. **Preprocessing:** Image resized to 640×480, normalized to [0,1]

2. **Encoding:** CNN extracts spatial features from image

3. **Attention:** Decoder learns which image regions to focus on when generating each token

4. **Decoding:** LSTM generates LaTeX tokens one at a time:
   - Start with `<SOS>` token
   - At each step: feed previous token + encoder output + attention weights
   - Generate next token (greedy: argmax of logits)
   - Stop at `<EOS>` token or max length

5. **Expression Parsing:** LaTeX → Python expression (e.g., `\times` → `*`)

6. **Safe Evaluation:** Use `eval()` with restricted builtins + AST whitelist
   - No imports, no function calls
   - Only arithmetic operators allowed
   - Handles ZeroDivisionError gracefully

## Example

**Input image:** Handwritten equation "2 + 3 × 4"

**Model output:**
```
LaTeX: 2 + 3 \times 4
Expression: 2+3*4
Answer: 14
```

## Performance

Trained on 10k MathWriting samples:
- Vocabulary size: ~500 LaTeX tokens
- Model parameters: ~39.5M
- Training time: ~2 hours on CPU, ~20 min on GPU

**Note:** Early epochs output garbage (model warming up). Performance improves significantly after 3-5 epochs.

## Dataset

Uses **MathWriting 2024** dataset (230k human-written + 400k synthetic handwritten equations):
- INKML format (XML with stroke coordinates and LaTeX ground truth)
- Train/valid/test splits
- Real handwriting with various styles and complexities

Data is not included in this repo. Download separately if training from scratch.

## Limitations & Future Work

- **Current:** Model trained on limited samples; predictions sometimes inaccurate
- **Segmentation:** End-to-end avoids symbol segmentation issues, but trades interpretability
- **Scope:** Handles basic arithmetic and fractions; complex multi-line equations may fail
- **Future:** Fine-tune on larger dataset, add beam search decoding, support more LaTeX commands

## References

- Seq2Seq: [Sutskever et al., 2014](https://arxiv.org/abs/1409.3215)
- Attention: [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473)
- MathWriting Dataset: [Malay et al., 2024](https://github.com/mathwriting/dataset)
