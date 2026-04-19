import cv2
import numpy as np


def preprocess_image(image_input):
    """
    Load and preprocess an equation image.
    Input: file path (str) or numpy array (BGR).
    Output: binary image (0=black, 255=white), THRESH_BINARY_INV applied.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_input}")
    else:
        img = image_input

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binary threshold with Otsu (BINARY_INV: dark ink on white paper -> white on black)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary


def get_bounding_boxes(binary_image, min_area=50):
    """Get bounding boxes of all symbols (sorted left-to-right)."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area:
            bboxes.append((x, y, w, h))

    # Try to merge '=' sign (two horizontal bars)
    bboxes = merge_equals_sign(bboxes)

    # Sort by x-coordinate (left-to-right)
    bboxes.sort(key=lambda b: b[0])

    return bboxes


def merge_equals_sign(bboxes):
    """Merge two horizontal bars into one bounding box if they form an '=' sign."""
    if len(bboxes) < 2:
        return bboxes

    merged = []
    used = set()

    for i, (x1, y1, w1, h1) in enumerate(bboxes):
        if i in used:
            continue

        # Look for a nearby box that could be the other bar of '='
        best_j = None
        for j, (x2, y2, w2, h2) in enumerate(bboxes):
            if j <= i or j in used:
                continue

            # Check if boxes are roughly aligned (overlap horizontally >50%)
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            total_x = max(x1 + w1, x2 + w2) - min(x1, x2)

            if total_x > 0 and overlap_x / min(w1, w2) > 0.5:
                # Check vertical distance (should be small, max 0.5*avg_height apart)
                avg_h = (h1 + h2) / 2
                v_dist = min(abs(y1 - (y2 + h2)), abs(y2 - (y1 + h1)))

                if v_dist < 0.5 * avg_h:
                    best_j = j
                    break

        if best_j is not None:
            # Merge boxes
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
            merged.append((x, y, w, h))
            used.add(best_j)
        else:
            merged.append((x1, y1, w1, h1))

    return merged


def segment_symbols(binary_image, min_area=50, padding=4):
    """
    Segment equation image into individual 28x28 symbol crops (sorted left-to-right).
    """
    bboxes = get_bounding_boxes(binary_image, min_area=min_area)

    symbol_crops = []
    for x, y, w, h in bboxes:
        # Extract region with padding
        y_min = max(0, y - padding)
        y_max = min(binary_image.shape[0], y + h + padding)
        x_min = max(0, x - padding)
        x_max = min(binary_image.shape[1], x + w + padding)

        crop = binary_image[y_min:y_max, x_min:x_max]

        # Resize to 28x28
        crop_resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)

        symbol_crops.append(crop_resized)

    return symbol_crops


def segment_from_file(path, min_area=50, padding=4):
    """Convenience: preprocess image file and segment symbols."""
    binary = preprocess_image(path)
    return segment_symbols(binary, min_area=min_area, padding=padding)
