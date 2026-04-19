import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, random_split, DataLoader

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=']


def load_mnist_digits():
    """Load MNIST dataset using torchvision."""
    try:
        os.makedirs('datasets', exist_ok=True)
        # Download and load MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(root='datasets', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='datasets', train=False, download=True, transform=transform)

        # Convert to numpy
        x_train = (train_data.data.numpy() / 255.0).astype(np.float32)
        y_train = train_data.targets.numpy()

        x_test = (test_data.data.numpy() / 255.0).astype(np.float32)
        y_test = test_data.targets.numpy()

        # Expand dims for channel (H, W) -> (H, W, 1)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        return x_train, y_train, x_test, y_test

    except Exception as e:
        print("ERROR: Could not load MNIST dataset.")
        print("Download manually from:")
        print("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
        print("Or use torchvision: python -c \"from torchvision import datasets; datasets.MNIST(root='datasets', train=True, download=True)\"")
        print(f"\nOriginal error: {e}")
        sys.exit(1)


def generate_operator_images(samples_per_class=2000):
    """Generate synthetic operator images (28x28 grayscale white-on-black)."""
    operators = ['+', '-', '*', '/', '=']

    # Font candidates to try
    font_candidates = [
        '/System/Library/Fonts/DejaVuSans.ttf',  # macOS
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
        'C:\\Windows\\Fonts\\Arial.ttf',  # Windows
    ]

    def get_font(size):
        """Try to load a TTF font, fallback to default."""
        for font_path in font_candidates:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                pass
        return ImageFont.load_default()

    def create_operator_image(symbol, augment=True):
        """Create single augmented operator image."""
        img = Image.new('L', (28, 28), color=0)  # black background
        draw = ImageDraw.Draw(img)

        # Random font size
        font_size = np.random.randint(16, 25)
        font = get_font(font_size)

        # Get text bounding box and center it
        bbox = draw.textbbox((0, 0), symbol, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (28 - text_width) // 2 - bbox[0]
        y = (28 - text_height) // 2 - bbox[1]

        # Draw white symbol
        draw.text((x, y), symbol, fill=255, font=font)

        if augment:
            # Random rotation
            angle = np.random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=0, expand=False)

            # Random scale
            scale = np.random.uniform(0.78, 1.0)  # 22-28 px
            new_size = int(28 * scale)
            img = img.resize((new_size, new_size), Image.Resampling.LANCZOS)

            # Pad back to 28x28
            padded = Image.new('L', (28, 28), color=0)
            offset = (28 - new_size) // 2
            padded.paste(img, (offset, offset))
            img = padded

            # Random Gaussian noise
            img_array = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 10, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            img = Image.fromarray(img_array.astype(np.uint8), mode='L')

        return np.array(img, dtype=np.uint8)

    data = {}
    for operator in operators:
        images = [create_operator_image(operator) for _ in range(samples_per_class)]
        data[operator] = np.array(images)

    return data


def build_dataset(operator_samples=2000):
    """Build combined MNIST + synthetic operators dataset."""
    print("Loading MNIST digits...")
    x_train, y_train, x_test, y_test = load_mnist_digits()

    print(f"Generating {operator_samples} synthetic images per operator...")
    operator_data = generate_operator_images(samples_per_class=operator_samples)

    # Prepare operator data: split into train/test (80/20)
    operator_images = []
    operator_labels = []
    operator_test_images = []
    operator_test_labels = []

    for class_idx, operator in enumerate(['+', '-', '*', '/', '=']):
        images = operator_data[operator]
        n = len(images)
        split = int(0.8 * n)

        # Training samples
        operator_images.extend(images[:split])
        operator_labels.extend([class_idx + 10] * split)  # Classes 10-14 for operators

        # Test samples
        operator_test_images.extend(images[split:])
        operator_test_labels.extend([class_idx + 10] * (n - split))

    # Convert operator test images to match x_test shape (add channel dim)
    operator_test_images = np.array(operator_test_images) / 255.0
    operator_test_images = np.expand_dims(operator_test_images, axis=-1)  # Add channel dim

    # Combine test sets
    x_test = np.vstack([x_test, operator_test_images])
    y_test = np.hstack([y_test, np.array(operator_test_labels)])

    # Combine MNIST train with operator train
    operator_images_arr = np.array(operator_images) / 255.0
    operator_images_arr = np.expand_dims(operator_images_arr, axis=-1)  # Add channel dim
    x_train_combined = np.vstack([x_train, operator_images_arr])
    y_train_combined = np.hstack([y_train, np.array(operator_labels)])

    # Convert to torch tensors (N, 1, 28, 28) for CNN
    x_train_tensor = torch.from_numpy(x_train_combined).permute(0, 3, 1, 2).float()
    y_train_tensor = torch.from_numpy(y_train_combined).long()

    x_test_tensor = torch.from_numpy(x_test).permute(0, 3, 1, 2).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    print(f"\nDataset Summary:")
    print(f"Training set: {x_train_tensor.shape}")
    print(f"Test set: {x_test_tensor.shape}")
    print(f"\nClass distribution (training):")
    for class_idx in range(15):
        count = (y_train_combined == class_idx).sum()
        print(f"  {CLASS_NAMES[class_idx]:>2} (class {class_idx:2d}): {count:5d} samples")

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor
