#!/usr/bin/env python3
"""Equation recognition and solver (seq2seq end-to-end model)."""
import argparse
import os
import glob
import torch
import numpy as np
from pathlib import Path


def predict_equation(image_path, model, vocab, idx2token, device='cpu'):
    """Predict equation from image using seq2seq model."""
    from equation_solver.mathwriting_loader import decode_sequence, parse_inkml_strokes, strokes_to_image
    from PIL import Image

    try:
        if image_path.endswith('.inkml'):
            strokes, _ = parse_inkml_strokes(image_path)
            if not strokes:
                return None
            img = strokes_to_image(strokes)
            img_array = np.array(img, dtype=np.float32) / 255.0
        else:
            img = Image.open(image_path).convert('L')
            if img.size != (640, 480):
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0

        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_indices = model.generate(img_tensor, vocab, max_len=300, device=device)

        tokens = decode_sequence(generated_indices[0], idx2token)
        return ''.join(tokens)

    except Exception as e:
        return None


def parse_and_evaluate(latex_str):
    """Convert LaTeX to expression and evaluate."""
    if not latex_str:
        return None, None

    import re
    expr = latex_str.replace('\\times', '*')
    expr = expr.replace('\\cdot', '*')
    expr = expr.replace('\\div', '/')
    expr = expr.replace(' ', '')
    expr = re.sub(r'[^0-9+\-*/.=()]', '', expr)

    if not expr or '=' in expr:
        return expr, None

    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return expr, result
    except:
        return expr, None


def main():
    parser = argparse.ArgumentParser(description='Equation recognition and solver')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--dir', default='@equations', help='Equation images directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', default='models/seq2seq_mathwriting.pt', help='Model path')

    args = parser.parse_args()

    print("=" * 70)
    print("Handwritten Equation Recognition & Solver")
    print("=" * 70)

    device = torch.device(args.device)
    print(f"\nDevice: {device}")

    # Load model
    if not os.path.exists(args.model):
        print(f"\nERROR: Model not found at {args.model}")
        print("Train with: python3 train_mathwriting.py --train-samples 10000")
        return

    print(f"Loading model: {args.model}")
    from equation_solver.seq2seq_model import load_model
    model, vocab, idx2token = load_model(args.model, device=device)
    model.eval()

    # Process images
    if args.image:
        print(f"\nProcessing: {args.image}\n")

        latex = predict_equation(args.image, model, vocab, idx2token, device)
        if latex:
            print(f"Equation:  {latex}")
            expr, result = parse_and_evaluate(latex)
            if expr:
                print(f"Expression: {expr}")
            if result is not None:
                print(f"Answer:    {result}")
        else:
            print("Failed to recognize equation")

    else:
        if not os.path.exists(args.dir):
            os.makedirs(args.dir, exist_ok=True)
            print(f"\nCreated {args.dir}/ — add equation images there")
            return

        images = sorted(glob.glob(os.path.join(args.dir, '*')))
        images = [f for f in images if f.endswith(('.png', '.jpg', '.jpeg', '.inkml'))]

        if not images:
            print(f"\nNo images in {args.dir}")
            return

        print(f"\nProcessing {len(images)} images from {args.dir}...\n")

        for image_path in images:
            basename = Path(image_path).name
            print(f"→ {basename}")

            latex = predict_equation(image_path, model, vocab, idx2token, device)
            if latex:
                print(f"  Equation:  {latex}")
                expr, result = parse_and_evaluate(latex)
                if result is not None:
                    print(f"  Answer:    {result}")
            else:
                print(f"  Failed to recognize")

            print()


if __name__ == '__main__':
    main()
