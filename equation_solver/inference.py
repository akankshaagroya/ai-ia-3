import numpy as np
import torch
from . import segmentation, evaluator, model

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=']
MODEL_PATH = 'models/equation_classifier.pt'


def predict_symbol(clf, symbol_image, device='cpu'):
    """
    Predict a single symbol (28x28 grayscale image).
    Returns: (symbol_name, confidence_score).
    """
    # Normalize and prepare tensor
    x = symbol_image.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)  # Add batch dimension (1, 28, 28)
    x = np.expand_dims(x, axis=0)  # Add channel dimension (1, 1, 28, 28)
    x_tensor = torch.from_numpy(x).to(device)

    # Predict
    with torch.no_grad():
        outputs = clf(x_tensor)
        probs = torch.softmax(outputs, dim=1)
        class_idx = torch.argmax(probs[0]).item()
        confidence = float(probs[0][class_idx].item())

    return CLASS_NAMES[class_idx], confidence


def predict_equation(clf, image_path, device='cpu'):
    """
    Predict the entire equation from an image file.
    Returns: (equation_string, [(symbol, confidence), ...]).
    """
    # Segment symbols
    symbol_images = segmentation.segment_from_file(image_path)

    if not symbol_images:
        return "", []

    # Predict each symbol
    predictions = []
    equation_chars = []

    for symbol_img in symbol_images:
        symbol_name, conf = predict_symbol(clf, symbol_img, device=device)
        predictions.append((symbol_name, conf))
        equation_chars.append(symbol_name)

    equation_string = ''.join(equation_chars)

    return equation_string, predictions


def run_pipeline(image_path, clf=None, device='cpu'):
    """
    Full end-to-end pipeline: image -> equation -> result.
    Returns dict with image_path, symbols, equation, expression, result.
    """
    if clf is None:
        clf = model.load_model(MODEL_PATH, device=device)

    equation_string, predictions = predict_equation(clf, image_path, device=device)

    expression = evaluator.extract_expression(equation_string)
    result = evaluator.evaluate_expression(equation_string)

    return {
        'image_path': image_path,
        'symbols': predictions,
        'equation': equation_string,
        'expression': expression,
        'result': result,
    }
