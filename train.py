#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from equation_solver import data_loader, model


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def eval_epoch(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train handwritten equation recognition model')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--operator-samples', type=int, default=1500, help='Synthetic images per operator')
    parser.add_argument('--model-path', default='models/equation_classifier.pt', help='Path to save model')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split fraction')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()

    print("=" * 60)
    print("Handwritten Equation Recognition - Training (PyTorch)")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load and build dataset
    print("\n[1/4] Building dataset...")
    X_train, y_train, X_test, y_test = data_loader.build_dataset(
        operator_samples=args.operator_samples
    )

    # Split training set into train/val
    train_size = int((1 - args.val_split) * len(X_train))
    val_size = len(X_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        TensorDataset(X_train, y_train),
        [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    print("\n[2/4] Building model...")
    clf = model.build_model()
    clf.to(device)

    print("\nModel architecture:")
    print(clf)
    print(f"\nTotal parameters: {sum(p.numel() for p in clf.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    # Training loop
    print("\n[3/4] Training model...")
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(clf, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(clf, val_loader, criterion, device)

        print(f"Epoch {epoch+1:2d}/{args.epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        # Early stopping and learning rate scheduling
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            model.save_model(clf, args.model_path)
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        scheduler.step(val_acc)

    # Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    test_loss, test_acc = eval_epoch(clf, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.model_path}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
