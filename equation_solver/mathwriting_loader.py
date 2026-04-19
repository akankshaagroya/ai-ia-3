"""Load and process MathWriting dataset for end-to-end model training."""
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import glob
import os
import re
from collections import Counter


def parse_inkml_strokes(inkml_path):
    """Extract strokes from INKML file."""
    try:
        tree = ET.parse(inkml_path)
        root = tree.getroot()
        ns = {'ink': 'http://www.w3.org/2003/InkML'}

        # Get normalized label
        label = None
        for annotation in root.findall('ink:annotation[@type="normalizedLabel"]', ns):
            label = annotation.text
            break

        # Get strokes
        strokes = []
        for trace in root.findall('ink:trace', ns):
            points_str = trace.text
            if not points_str:
                continue
            points = []
            for point_pair in points_str.split(','):
                coords = point_pair.strip().split()
                if len(coords) >= 2:  # x y [time]
                    try:
                        x, y = float(coords[0]), float(coords[1])
                        points.append((x, y))
                    except ValueError:
                        continue
            if points:
                strokes.append(points)

        return strokes, label
    except Exception as e:
        return None, None


def strokes_to_image(strokes, width=640, height=480, line_width=2):
    """Render strokes to grayscale image."""
    if not strokes:
        return None

    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        if len(stroke) > 1:
            for i in range(len(stroke) - 1):
                x1, y1 = stroke[i]
                x2, y2 = stroke[i + 1]
                x1 = max(0, min(width - 1, int(x1)))
                y1 = max(0, min(height - 1, int(y1)))
                x2 = max(0, min(width - 1, int(x2)))
                y2 = max(0, min(height - 1, int(y2)))
                draw.line([(x1, y1), (x2, y2)], fill=0, width=line_width)

    return img


def load_mathwriting_samples(split='train', max_samples=None):
    """Load MathWriting samples and return (images, labels).

    Args:
        split: 'train', 'valid', or 'test'
        max_samples: Limit number of samples (for quick testing)

    Returns:
        List of (image_array, latex_label) tuples
    """
    base_path = f'datasets/mathwriting-2024/{split}'

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"MathWriting {split} split not found at {base_path}")

    inkml_files = sorted(glob.glob(os.path.join(base_path, '*.inkml')))
    if max_samples:
        inkml_files = inkml_files[:max_samples]

    samples = []
    skipped = 0

    print(f"Loading {len(inkml_files)} {split} samples...")

    for i, inkml_path in enumerate(inkml_files):
        strokes, label = parse_inkml_strokes(inkml_path)

        if not strokes or not label:
            skipped += 1
            continue

        img = strokes_to_image(strokes)
        if img is None:
            skipped += 1
            continue

        img_array = np.array(img, dtype=np.float32) / 255.0
        samples.append((img_array, label))

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(inkml_files)}: {len(samples)} valid samples (skipped {skipped})")

    print(f"Loaded {len(samples)} samples from {split} split (skipped {skipped})")
    return samples


def build_vocabulary(samples_list):
    """Build vocabulary from all LaTeX expressions.

    Args:
        samples_list: List of (image, label) tuples or multiple such lists

    Returns:
        vocab (dict): token -> index
        idx2token (dict): index -> token
    """
    all_tokens = Counter()

    # Ensure samples_list is a list of lists
    if samples_list and isinstance(samples_list[0], tuple):
        samples_list = [samples_list]

    for samples in samples_list:
        for img, label in samples:
            # Tokenize LaTeX: split by braces and backslashes
            tokens = tokenize_latex(label)
            all_tokens.update(tokens)

    # Create vocab with special tokens
    vocab = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }

    idx = 4
    for token, count in all_tokens.most_common():
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    idx2token = {v: k for k, v in vocab.items()}

    print(f"Vocabulary size: {len(vocab)}")
    return vocab, idx2token


def tokenize_latex(latex_str):
    """Tokenize LaTeX expression into meaningful tokens."""
    if not latex_str:
        return []

    tokens = []
    i = 0
    while i < len(latex_str):
        if latex_str[i] == '\\':
            # LaTeX command
            j = i + 1
            while j < len(latex_str) and latex_str[j].isalpha():
                j += 1
            tokens.append(latex_str[i:j])
            i = j
        elif latex_str[i] in '{}^_':
            # Special characters
            tokens.append(latex_str[i])
            i += 1
        elif latex_str[i] == ' ':
            # Skip spaces
            i += 1
        else:
            # Regular character or digit
            tokens.append(latex_str[i])
            i += 1

    return tokens


def encode_sequence(tokens, vocab, max_len=300):
    """Convert token sequence to indices."""
    indices = [vocab.get('<SOS>', 1)]
    for token in tokens:
        idx = vocab.get(token, vocab.get('<UNK>', 3))
        indices.append(idx)
    indices.append(vocab.get('<EOS>', 2))

    # Pad or truncate
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices = indices + [vocab.get('<PAD>', 0)] * (max_len - len(indices))

    return indices


def decode_sequence(indices, idx2token):
    """Convert indices back to tokens."""
    tokens = []
    for idx in indices:
        if idx == 0:  # PAD
            break
        token = idx2token.get(idx, '<UNK>')
        if token in ['<SOS>', '<EOS>', '<PAD>']:
            continue
        tokens.append(token)
    return tokens
