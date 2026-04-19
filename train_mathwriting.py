#!/usr/bin/env python3
"""Train seq2seq model on MathWriting dataset."""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from equation_solver.mathwriting_loader import (
    load_mathwriting_samples,
    build_vocabulary,
    tokenize_latex,
    encode_sequence,
)
from equation_solver.seq2seq_model import build_seq2seq_model, save_model


class MathWritingDataset(Dataset):
    """Dataset for MathWriting samples."""

    def __init__(self, samples, vocab, max_seq_len=300):
        self.samples = samples
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_array, latex_label = self.samples[idx]

        # Convert image to tensor (1, 640, 480)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()

        # Tokenize and encode LaTeX
        tokens = tokenize_latex(latex_label)
        token_indices = encode_sequence(tokens, self.vocab, max_len=self.max_seq_len)
        token_tensor = torch.tensor(token_indices, dtype=torch.long)

        return img_tensor, token_tensor


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    batch_count = 0

    for i, (images, token_ids) in enumerate(train_loader):
        images = images.to(device)
        token_ids = token_ids.to(device)

        optimizer.zero_grad()

        # Encode images
        encoder_output = model.encode(images)

        # Decode with teacher forcing
        max_seq_len = token_ids.size(1) - 1
        logits = model.decode_batch(encoder_output, token_ids[:, :-1], max_len=max_seq_len)

        # Compute loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_reshaped = logits.reshape(batch_size * seq_len, vocab_size)
        targets_reshaped = token_ids[:, 1:seq_len+1].reshape(batch_size * seq_len)  # Shift targets

        loss = criterion(logits_reshaped, targets_reshaped)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}: loss={loss.item():.4f}")

    avg_loss = total_loss / batch_count
    return avg_loss


def eval_epoch(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for images, token_ids in val_loader:
            images = images.to(device)
            token_ids = token_ids.to(device)

            encoder_output = model.encode(images)
            max_seq_len = token_ids.size(1) - 1
            logits = model.decode_batch(encoder_output, token_ids[:, :-1], max_len=max_seq_len)

            batch_size, seq_len, vocab_size = logits.shape
            logits_reshaped = logits.reshape(batch_size * seq_len, vocab_size)
            targets_reshaped = token_ids[:, 1:seq_len+1].reshape(batch_size * seq_len)

            loss = criterion(logits_reshaped, targets_reshaped)
            total_loss += loss.item()
            batch_count += 1

    avg_loss = total_loss / batch_count
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train seq2seq on MathWriting')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train-samples', type=int, default=50000, help='Max training samples')
    parser.add_argument('--val-samples', type=int, default=5000, help='Max validation samples')
    parser.add_argument('--model-path', default='models/seq2seq_mathwriting.pt', help='Model save path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("=" * 70)
    print("Training Seq2Seq Model on MathWriting")
    print("=" * 70)

    device = torch.device(args.device)
    print(f"\nDevice: {device}")

    # Load samples
    print("\n[1/4] Loading training samples...")
    train_samples = load_mathwriting_samples(split='train', max_samples=args.train_samples)

    print("\n[2/4] Loading validation samples...")
    val_samples = load_mathwriting_samples(split='valid', max_samples=args.val_samples)

    # Build vocabulary
    print("\n[3/4] Building vocabulary...")
    vocab, idx2token = build_vocabulary([train_samples, val_samples])

    # Create datasets and loaders
    train_dataset = MathWritingDataset(train_samples, vocab)
    val_dataset = MathWritingDataset(val_samples, vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Build model
    print("\n[4/4] Building model...")
    model = build_seq2seq_model(len(vocab), device=args.device)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    # Training loop
    print("\nTraining...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, vocab, idx2token, args.model_path)
            print(f"  Checkpoint saved!")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step(val_loss)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Model saved to: {args.model_path}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
