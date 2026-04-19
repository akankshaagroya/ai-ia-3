#!/usr/bin/env python3
import argparse
import os
import glob
import torch

from equation_solver import inference, model, segmentation
import cv2
import numpy as np


def print_result(result, confidence_threshold=0.5):
    """Pretty-print inference result."""
    print(f"\nImage: {result['image_path']}")
    print("-" * 60)

    # Print detected symbols with confidence
    if result['symbols']:
        symbol_str = " ".join([f"{sym}({conf*100:.1f}%)" for sym, conf in result['symbols']])
        print(f"Detected symbols: {symbol_str}")

        # Warn on low confidence
        for sym, conf in result['symbols']:
            if conf < confidence_threshold:
                print(f"  ⚠️  Low confidence on '{sym}' ({conf*100:.1f}%)")
    else:
        print("Detected symbols: (none)")

    print(f"Equation:  {result['equation']}")
    print(f"Expression: {result['expression']}")

    # Print result
    if isinstance(result['result'], str) and result['result'].startswith('ERROR'):
        print(f"Result:    {result['result']}")
    else:
        print(f"Result:    {result['result']}")

    print("-" * 60)


def visualize_segmentation(image_path, device, model_path='models/equation_classifier.pt'):
    """Show image with bounding boxes for each segmented symbol."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not available for visualization")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get segmentation info
    binary = segmentation.preprocess_image(image_path)
    bboxes = segmentation.get_bounding_boxes(binary)

    # Load model and predict
    clf = model.load_model(model_path, device=device)
    _, predictions = inference.predict_equation(clf, image_path, device=device)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(img_rgb)

    # Draw bounding boxes and labels
    for (x, y, w, h), (symbol, conf) in zip(bboxes, predictions):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        label = f"{symbol} ({conf*100:.0f}%)"
        ax.text(x, y - 5, label, fontsize=10, color='lime', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax.set_title(f"Equation Segmentation: {predictions}")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test handwritten equation recognition model')
    parser.add_argument('--image', help='Path to single test image')
    parser.add_argument('--dir', help='Directory of test images')
    parser.add_argument('--model-path', default='models/equation_classifier.pt', help='Path to trained model')
    parser.add_argument('--visualize', action='store_true', help='Show bounding boxes')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Warn if symbol confidence < this')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine image paths
    image_paths = []

    if args.image:
        image_paths = [args.image]
    elif args.dir:
        pattern = os.path.join(args.dir, '**', '*')
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(glob.glob(os.path.join(args.dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.dir, '**', ext), recursive=True))
        image_paths = list(set(image_paths))  # Remove duplicates
    else:
        default_dir = 'datasets/test_equations'
        pattern = os.path.join(default_dir, '**', '*')
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(glob.glob(os.path.join(default_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(default_dir, '**', ext), recursive=True))
        image_paths = list(set(image_paths))

    if not image_paths:
        print("=" * 60)
        print("No images found.")
        print(f"Add test images to 'datasets/test_equations/' or specify with --image/--dir")
        print("=" * 60)
        return

    print("=" * 60)
    print("Handwritten Equation Recognition - Inference (PyTorch)")
    print("=" * 60)
    print(f"Using device: {device}\n")

    # Load model once
    try:
        clf = model.load_model(args.model_path, device=device)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run 'python train.py' first to train the model.")
        return

    # Process images
    image_paths.sort()
    for image_path in image_paths:
        try:
            result = inference.run_pipeline(image_path, clf=clf, device=device)
            print_result(result, confidence_threshold=args.confidence_threshold)

            if args.visualize:
                visualize_segmentation(image_path, device, args.model_path)

        except Exception as e:
            print(f"\nERROR processing {image_path}: {e}")
            print("-" * 60)

    print("\n" + "=" * 60)
    print(f"Processed {len(image_paths)} image(s)")
    print("=" * 60)


if __name__ == '__main__':
    main()
